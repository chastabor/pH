"""`ctx.workspace` — where an agent's writes land, and how honestly that is stated.

The seam the containment ladder hangs off (D21, §4.8). Its consumer is the
**agent lifecycle**, not a tool: an agent acquires a workspace, and `ctx.fs`'s
root and `ctx.subprocess`'s cwd resolve to `workspace.root`, which is what makes
a tier bound *authored* code rather than merely observe it.

Invariants this seam holds:

* **`repo_writable` records which guarantee was obtained, never which was
  requested.** A caller asking `access="read"` gets the strongest kind the
  mounted tier can actually provide; `False` means a tier is enforcing it. Any
  wording here, in `ph doctor`, or in a config comment that blurs request and
  guarantee is a defect (§12 Q10).
* **There is always a workspace.** `acquire` never fails and never returns
  `None`. A provider that cannot serve a request *declines*, and the seam falls
  back to `shared` with a logged notice.
* **A workspace is an effect of the scope that took it (I2).** `acquire`
  registers its teardown through `ctx.effect`, so a disposed agent scope unwinds
  the workspace. That is the in-process half of cleanup; the
  `workspace/acquired`/`disposed` pair is the crash half, reconciled at session
  open (§4.9).
* **`scratch` is always present and always writable**, on every kind and every
  tier, and the *seam* creates it — one implementation rather than one per
  provider. It lives in pH's own state directory rather than in the workspace,
  so it survives disposal as a session artifact. A provider is handed the path
  and may substitute its own, but never has to invent the layout.
* **The kind predicates below are exhaustive `match`es, never membership tests**,
  so a seventh `WorkspaceKind` fails to type-check rather than silently
  classifying.

@module ph.seams.workspace
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import anyio
from pydantic import Field

from ..agent.types import AgentHandle, PreStepDecision, PreStepRequest
from ..cordis import Context, Disposer, Next, Running, maybe_await, plugin, running, safe_yaml_load
from ..json import as_str
from ..keys import AGENTS, CONTAINMENT, FS, SESSION_PERSISTENCE, SESSIONS, TOOLS, WORKSPACE
from ..paths import canonical, default_home_path
from ..session import Session, SessionEvent
from ..tools.definition import ToolExecution, ToolExecutionResult
from ..tools.errors import HarnessError
from ..wire import WireModel, literal_lookup
from . import workspace_provision
from ._registry import claim_entry, claim_slot
from .diagnostics import Diagnostic, contribute
from .sandbox import SandboxPolicy
from .subagents import descendants
from .telemetry import ops_record
from .workspace_provision import ProvisionEntry, ProvisionReport

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
    "apply",
    "checkpoint_policy",
    "checkpoints",
    "discards_writes",
    "discover_provisioning",
    "family_survivors",
    "fresh_root",
    "latest_checkpoint",
    "lifecycle",
    "measure_strays",
    "project_access",
    "redirection_env",
    "sanitize_ref",
    "stored_survivors",
    "workspace_leaks",
    "workspace_of",
    "workspace_policy",
    "workspace_survivors",
    "writable_roots",
]

log = logging.getLogger("ph.seams.workspace")

ContainmentTier: TypeAlias = Literal["advisory", "worktree", "sandbox"]
"""The ladder, named once: `ph doctor` prints it (P4-12) and `containment.tier`
selects it (P4-11).
"""

WorkspaceKind: TypeAlias = Literal[
    "shared",
    "worktree",
    "worktree-ephemeral",
    "readonly-scratch",
    "overlay",
    "overlay-ephemeral",
]
"""What an agent actually got.

* `shared` — one checkout, no isolation. The only kind whose root *is* the base.
* `worktree` — that agent's own branch, merged back deliberately.
* `worktree-ephemeral` — a full checkout the agent may write and whose writes
  **reach nobody**: discarded on disposal, never merged.
* `readonly-scratch` — the repository is genuinely unwritable. Only a sandbox
  backend can deliver it.
* `overlay` / `overlay-ephemeral` — copy-on-write *views* of the tree: the agent
  may write anywhere and the host sees none of it, because every change lands in
  a delta layer. They split as the worktree kinds do: `overlay` keeps its delta
  at release so the work can be exported onto a branch, `overlay-ephemeral`
  throws it away.

An overlay is **not** a flavour of `worktree-ephemeral`: it contains the tree as
it is, untracked and ignored files included, and its history is not git's, so
`workspace_git` must decline it rather than run `write-tree` against a mountpoint.
"""

WorkspaceAccess: TypeAlias = Literal["write", "read"]
"""What the *caller* needs of `base` — a request, not a guarantee. The tier decides
whether a read-only claim can be enforced; the answer comes back as `kind` and
`repo_writable`.
"""


def project_access(kind: WorkspaceKind) -> WorkspaceAccess:
    """What a workspace of this kind grants of the **project** (E3).

    Not of the directory: `worktree-ephemeral` may be written freely and merges
    nothing, so what its holder was granted of the project is `read`. Recorded by a
    spawn as `granted_access` and printed per agent by `ph doctor`.
    """
    match kind:
        case "shared" | "worktree" | "overlay":
            # `overlay`: its delta survives release and can be exported onto a
            # branch, so what the holder wrote can reach the project. Deliberate,
            # exactly as a merge is — granting write promises nobody ran it.
            return "write"
        case "worktree-ephemeral" | "readonly-scratch" | "overlay-ephemeral":
            return "read"


def fresh_root(kind: WorkspaceKind) -> bool:
    """Whether this kind hands the agent a directory that is not the base.

    `shared` is the only one whose root *is* the base, which is why nothing is
    provisioned into it: every material is already there, and copying `.env` onto
    itself would destroy the file the provisioning exists to provide.
    """
    match kind:
        case "shared":
            return False
        case (
            "worktree" | "worktree-ephemeral" | "readonly-scratch" | "overlay" | "overlay-ephemeral"
        ):
            return True


def discards_writes(kind: WorkspaceKind) -> bool:
    """Whether release throws the agent's writes away, dirty tree and all (P6-28).

    The predicate the retention policy keys on. The kinds answering `True` are the
    only ones whose *evidence* an ordinary release can lose, which is what makes
    them the only ones a policy has any business retaining by default.
    """
    match kind:
        case "shared" | "worktree" | "readonly-scratch" | "overlay":
            return False
        case "worktree-ephemeral" | "overlay-ephemeral":
            return True


def redirection_env(scratch: Path) -> dict[str, str]:
    """Where the toolchain's droppings go instead of into the workspace (E12).

    Every entry is a cache or temp location a build tool writes *beside the sources*
    by default, pointed inside `scratch` — which is outside the workspace and
    survives disposal. At the `worktree` tier this is what makes `git status` report
    the agent's work rather than `pytest`'s, so "remove a clean worktree, keep a
    dirty one" keeps meaning something.

    A property of *scratch*, not of worktrees, so it lives beside `Workspace.env`
    rather than in the git tier that first needed it; §4.8 gives the same env to
    `readonly-scratch`.

    `PYTEST_ADDOPTS` disables the cache provider outright as well as moving
    `--basetemp`, because `.pytest_cache/` is written next to `rootdir` and no
    environment variable relocates it. `TMPDIR` is `scratch` itself: it must exist
    before the first `tempfile` call, and `scratch` is the one directory the seam
    guarantees.
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

    The one definition of the set, because `permissions-fs`'s default rule prompts
    about what falls outside it and `workspace_policy` hands the same set to a
    backend to enforce; two spellings that drifted would be a tier whose name
    promises what its policy does not do.

    `scratch` is always in it. It is outside the worktree by design (E5) and is the
    one place a read-only or ephemeral agent is *told* it may write, so a set naming
    only `root` would prompt on exactly the writes the design invites.

    **Canonical, by construction rather than by resolving here.** The kernel matches
    the path it resolves — Seatbelt refused a workspace spelled `/var/folders/…` its
    own writes, because that is `/private/var/…` to the kernel — and a backend that
    re-spelled the set privately would have made the enforced boundary and the
    prompted one two different strings. So the seam canonicalises every input it
    mints roots from (`acquire`), and this returns what the seam handed out.
    """
    return (workspace.root, workspace.scratch)


def workspace_policy(workspace: Workspace) -> SandboxPolicy:
    """The workspace as a confinement request: write here, ask about elsewhere.

    Derived from `writable_roots` rather than restating it, so `ctx.shell`'s
    enforced boundary and `workspace-write-scope`'s prompt boundary cannot drift.
    """
    first, *extra = writable_roots(workspace)
    return SandboxPolicy(
        mode="workspace-write",
        workspace_root=str(first),
        writable_extra=[str(path) for path in extra],
    )


def workspace_of(ctx: Context, agent: AgentHandle | str | None) -> Workspace | None:
    """This agent's workspace, asked of a seam that may not be mounted.

    **A handle, a bare id, or nothing.** The body always accepted all three —
    `rlm-prompt` passes the id it was given, the fs resolver passes whatever it
    holds — and `Any` was hiding that this is one function answering for two
    calling conventions (plan P8-03, issue 3).

    The question written once, for the five callers that had it: the prompt line,
    `bash`, the kernel, the spawn path and the fs resolver.

    **Fail-soft on purpose.** A caller asking "where does this agent write" during a
    teardown, or in a profile that layers no workspace row, gets `None` and carries
    on with the process's own directory — a raising seam here would make an absent
    optional row fatal.
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
"""What every ref pH makes is named under.

Here rather than in the git tier that first wrote it, for the reason `CHECKPOINT`
moved: **it is the seam's ref namespace, not one tier's**. Three tiers and one
command have to agree on it — the jj tier names its bookmarks with it, the overlay
tier its exports, and `/workspaces` enumerates it — so leaving it in
`workspace_git` made it a vocabulary the others borrowed, and made a deployment
that never layers the git tier import it anyway for a string constant.


Shared with `/workspaces`, which enumerates the prefix to find the artifacts
disposal leaves. A management command that offered to delete a person's own
branches would be a different and much worse tool, and this prefix is the whole
of what keeps it from being one — so the two must not drift.
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


EXCLUDE = ":(exclude)"
"""Git's pathspec magic for "everything but this".

Named because `workspace_git._commit` reads it back off the pathspec to learn
which entries were provisioned: the two are one fact, and a marker spelled twice
is a secret that reaches a branch the day one of them changes.
"""


@dataclass(frozen=True, slots=True)
class Workspace:
    """One agent's working directory, and the truth about what it bounds.

    A value: what the provider decided, and nothing about who is holding it. The
    seam keeps the bookkeeping — which agent, which session, how to end it — so a
    provider cannot half-implement the lifecycle.
    """

    root: Path
    """The agent's cwd: `ctx.fs`'s root and `ctx.subprocess`'s default cwd."""
    scratch: Path
    """Always writable, on every kind and tier, and outside `root` on purpose —
    so it survives disposal even when the workspace itself is discarded. Handed
    to the provider already created; a provider substitutes only if its tier
    needs the path somewhere else."""
    kind: WorkspaceKind
    repo_writable: bool
    """Whether `root` can actually be written. **`False` only when a tier is
    enforcing it** — never as a statement of intent, and never inferred from
    `access`."""
    ref: str | None = None
    """The git branch, when the kind has one."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Environment a runner should apply, for a kind that needs redirecting.

    Empty for `shared`. The read-only kinds point `TMPDIR`, `PYTEST_ADDOPTS` and
    friends inside `scratch`, because build tools write into the tree they are run
    against. Best-effort by construction: a toolchain that insists on writing beside
    its sources will still fail, and the answer to that is `access="write"` for that
    agent, not a weaker tier.
    """
    provisioned: tuple[str, ...] = ()
    """Paths the seam put in this workspace (E14) — not the agent's work."""
    provision_failures: tuple[str, ...] = ()
    """Materials the seam could not put in place (E14).

    On the value rather than in a log line because the party that has to know
    `.env` is missing is the *agent* about to wonder why the tests fail — it is
    read straight onto the workspace prompt line. Empty is the ordinary case,
    including "this profile provisions nothing"."""
    retained: str = ""
    """Why this tree is being kept, set late by whoever learns the outcome (P6-28).

    Late state rather than an acquire-time field or a `release` argument: nobody
    knows at acquire how the child will end, and `release` runs as a *scope
    disposer*, so it is told nothing about why.

    A **reason**, not a flag, which is what lets the fold tell apart three states
    that all leave a tree on disk: a deliberate keep says why here, a dirty-tree
    keep is `kept` with no reason, and a leak has no `disposed` event at all.
    """
    release: Callable[[Workspace], Awaitable[bool]] | None = None
    """The provider's teardown, returning whether anything was **kept**.

    Takes the whole workspace, not one field: a teardown policy needs what was
    provisioned *and* what kind it holds, and P4-09's checkpoint refs will be a
    third such fact.

    The answer is only knowable here — P4-08's policy is "commit, then remove,
    discard ephemeral even if dirty", so `kind` cannot be asked instead. The seam
    records it on `workspace/disposed` so a reader can tell "nothing changed, so it
    was removed" from "these writes were thrown away by design".
    """

    def agent_work_pathspec(self) -> list[str]:
        """A `git` pathspec selecting this tree *minus* what the seam put in it.

        The one definition of "the agent's work", for three consumers that must not
        disagree: the disposal policy (`workspace_git._dirty`), `/workspaces list`, and
        P4-09's `/revert` — whose "restore tracked + untracked-not-ignored" is exactly
        the set that must not clobber a provisioned `node_modules`.

        The positive `.` is required: exclusions alone match everything, which is the
        opposite of what they read as.
        """
        return [".", *(f"{EXCLUDE}{entry}" for entry in self.provisioned)]


DeclineReason: TypeAlias = Literal[
    "not-a-repository",
    "branch-in-use",
    "path-exists",
    "provider-failed",
    "overlay-failed",
]
"""Why a tier could not serve a request, as a code rather than prose.

`ph doctor` prints it (P4-12). An operator who set `worktree` and got `shared`
is owed the reason, and a durable event carrying an English sentence is
unparseable by the consumer that has to branch on it.
"""


class WorkspaceDeclined(Exception):
    """A provider declining *with* a reason. Never fatal; the seam falls back.

    Raised rather than returned so the `acquire` protocol keeps one shape: a
    bare `None` is still a decline, it simply cannot say why.
    """

    def __init__(self, reason: DeclineReason, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason: DeclineReason = reason


class ChildWorkspaceMissing(HarnessError):
    """A child reached its first step holding no workspace. Refused, loudly (§6.5).

    **The lazy acquire below is for the person's own agent, and answering it for a
    child would widen that child.** A child's workspace is its *parent's* decision
    — the base to fork from and the `access` to grant both arrive with the spawn and
    are recorded on its admission — so the only thing this row could do for a child
    is invent them: the process's own directory as `base`, and whatever the profile
    configured for the *root* as `access`, which defaults to `write`. A research
    child admitted as `read` would come back holding a writable checkout of
    somebody else's tree, and every event about it would say so honestly while being
    wrong about what was asked for.

    So the absence is treated as the bug it is. A spawn path that forgot to acquire
    is a **missing call**, and the loud version costs a failed turn that names the
    child; the quiet version costs a child that holds more than its admission
    recorded, which nothing downstream can detect — `workspace/acquired` reports
    what it got, not what it should have got, and §6.5 is checked at admission where
    this never appeared.

    Not `WorkspaceDeclined`: that is a *tier* saying it cannot serve a request, and
    the seam is right to fall back for it. This is a caller asking a question that
    was never this row's to answer.

    A `HarnessError`, so the code survives the trip. Raised from `agent/pre-step`
    this becomes a recorded `turn/end` with `reason.kind: "error"` — loud, and
    durable — and `error_info` is what puts a name in it: a bare exception reaches
    that record as `UNKNOWN`, which is the prose-only failure the code exists to
    avoid. `HarnessError` sets `self.code`, which is also the attribute the daemon's
    `respond` reads, so one spelling serves both readers.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, "CHILD_WORKSPACE_MISSING")


@runtime_checkable
class WorkspaceProvider(Protocol):
    """A tier's implementation. `None` declines, and declining is normal.

    Typed rather than duck-typed, and `tier` is a member rather than a `getattr`
    probe: a provider whose method drifted would otherwise fail at runtime inside
    the seam's `except`, be reported to the operator as `shared`, and take the
    containment with it silently.

    `tier` is a property for `CodeRuntime`'s reason: every tier declares it as a
    frozen field, and a settable Protocol attribute refused all of them.
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


@runtime_checkable
class ReclaimingProvider(Protocol):
    """A provider that can release a workspace it did not create (F6).

    **An optional capability as its own Protocol, never a `getattr` probe** — the
    shape `ExportingProvider` and `RehydratableProvider` also take. Not every tier
    can reclaim (an in-memory one has nothing to), and a probe would report a
    provider whose method is *misnamed* as one that cannot, which hides a leak
    instead of closing it.

    Returns whether anything was **kept**, matching `Workspace.release`, so the
    `workspace/disposed` a reconciliation writes says what an orderly one would.
    """

    async def reclaim(self, record: WorkspaceRecord) -> bool: ...


@runtime_checkable
class ExportingProvider(Protocol):
    """A provider that can put an agent's work where the project can see it.

    An optional capability for `ReclaimingProvider`'s reason: `shared` has nothing
    to export, and a tier that isolates by *discarding* has nothing to offer.

    Returns the git ref the work is on, so one verb serves both isolating tiers — a
    worktree answers with the branch it has been committing to, an overlay builds
    one out of its delta first. `/workspaces` asks the seam rather than asking which
    tier it is talking to.
    """

    async def export(self, record: WorkspaceRecord) -> str: ...


@dataclass(frozen=True, slots=True)
class Stray:
    """One checkout a tier still has on disk, and what removing it would cost.

    Named for the interesting case rather than the common one: since disposal
    commits and takes the checkout back, a directory that is still there means a
    live agent is working in it or disposal could not remove it.
    """

    ref: str
    """The branch or bookmark this checkout is on — the key `/workspaces` joins on,
    because it is the one name every tier puts on a `Workspace` and carries rather
    than derives from a path."""
    path: Path
    dirty: bool
    """Whether removing the directory would lose something the ref does not have.

    **Each tier answers in its own terms and they are not the same question.** A git
    worktree is dirty when it holds uncommitted changes. A jj workspace's working
    copy *is* a commit and its bookmark follows it, so asking is also what makes the
    answer safe — a `True` there means "there is work, and it is now on the ref",
    which is a better answer than git can give.
    """


async def measure_strays(
    found: Sequence[tuple[str, Path]],
    probe: Callable[[Path], Awaitable[bool]],
    *,
    with_status: bool,
    skip: Container[str] = (),
) -> list[Stray]:
    """`(ref, path)` pairs into `Stray`s, measuring `dirty` concurrently.

    Beside `Stray` because it was the same twelve lines in both tiers — the
    `dict.fromkeys`, the `with_status` guard, the task group, the comprehension —
    differing only in the probe, which is the one part that is genuinely the tier's.
    A change to the fan-out or to `Stray` landed twice, in files that already share
    code deliberately.

    Concurrent because each probe is a subprocess, and a run that stranded one
    checkout has usually stranded several.

    **`skip` is the refs a live agent holds, and leaving it out cost more than
    time.** `/workspaces` never reads `dirty` for a held row — `describe` reports it
    as `held` and stops — so every probe against one was a subprocess spent on an
    answer nobody looks at, one per live child per listing. On the jj tier it was
    worse than waste: the probe snapshots, so *listing* mutated the working-copy
    commit of every agent that was mid-turn. The old command filtered these out and
    the filter did not survive being moved behind a Protocol; it lives here now,
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
    """A provider that can list, merge and delete the refs it leaves behind (E15).

    An optional capability as its own Protocol, `ReclaimingProvider`'s shape and for
    its reason. It exists because `/workspaces` was doing this itself, **in git** —
    `git branch --list ph/*`, `git merge`, `git branch -d` — and `jj` embeds its own
    git implementation. *Measured*: the whole acquire, bookmark and export flow runs
    with a `git` on `PATH` that exits 127 and the binary is never invoked. So a jj
    deployment need not have it, and there the branch listing failed and
    `/workspaces` answered "no agent workspaces are left behind" while the bookmarks
    were sitting right there.

    `refs` returns **everything**, not just what pH made: the prefix guard belongs to
    the command that would otherwise offer to delete somebody's `feature/x`, and the
    second caller is `merge`, which has to accept a ref outside the prefix.

    Both acting verbs return `""` for "it happened" and otherwise **the sentence a
    person reads**, rather than a bool: what went wrong is the tier's to say, and the
    two tiers do not fail the same way. A conflicted merge is the case that proves
    it — git exits non-zero and refuses, jj exits **zero** and records the conflict
    in the commit, so a caller testing an exit code reports a clean merge that a
    person then discovers by opening the file.
    """

    async def refs(self, base: Path) -> list[str]: ...

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str: ...

    async def merge(self, base: Path, ref: str) -> str: ...


@runtime_checkable
class EnumeratingProvider(Protocol):
    """A provider that can find and take back the checkouts it left on disk (E15).

    **Separate from `ArtifactProvider`, and the overlay tier is why.** These were one
    Protocol of five verbs, justified as "`/workspaces`' whole surface — a tier
    serves that command or it does not". `workspace-agentfs` falsified that in the
    same breath: its artifact is a git branch like anyone else's, so it answers the
    ref verbs, but a *mountpoint is not a checkout* this command can hand back — so
    it had to write two stubs to be admitted, and `isinstance` could not tell a tier
    that meant them from one that had merely typed them. Two Protocols along the line
    the code was already drawing, and the stubs are gone.

    `strays` filters by the provider's **own root**, which is the whole safety of
    `discard`: `jj workspace list` reports the person's own `default` workspace at
    the repository root, and a `remove` that trusted it would delete their
    repository. It is also what let the command stop carrying a copy of that setting.

    `discard` removes a checkout and nothing else — the ref is the artifact, and
    `/workspaces` deletes that separately, deliberately, behind its own flag.
    """

    async def strays(
        self, base: Path, *, with_status: bool = True, skip: Container[str] = ()
    ) -> list[Stray]: ...

    async def discard(self, path: Path) -> str: ...


@runtime_checkable
class CheckpointingProvider(Protocol):
    """A provider that can name a workspace's state and put it back (P4-09).

    An optional capability as its own Protocol — `ReclaimingProvider`'s shape, and
    for its reason. Not every tier has a restore mechanism: `shared` is the
    person's own checkout, and an overlay's delta is a perfectly good restore point
    that simply is not a git tree. A `getattr` probe would report a provider whose
    method drifted as one that cannot checkpoint, which loses `/revert` for a tier
    that has it, silently.

    **`capture` returns one token serving two readers**, which is the rule the git
    tier already argued for its tree hash: P4-09 stores it as a restore point and
    P5-07 uses it as a *fingerprint*, to decide that a quality gate which failed
    against this exact state need not run again. One derivation, so a gate memo and
    a restore point can never disagree about whether the work changed.

    What a token must promise is therefore narrow and exact: **equal tokens mean
    equal content**. The converse is *not* required — a tier whose token moves
    while the content stands still only makes a gate re-run, which is the safe
    direction — and `workspace-jj` is such a tier and says so in its own words.

    `capture` also **pins** what it names, for as long as the workspace lives. An
    unreferenced git tree is eligible for `gc`, and a restore point that evaporates
    is worse than none, because `/revert` listed it.
    """

    async def capture(self, workspace: Workspace) -> str | None: ...

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]: ...


Backend: TypeAlias = Literal["git", "jj", ""]
"""Which version control a tier is backed by, or `""` for none of them.

Here rather than in `ph.seams.changes`, which is the module that consumes it: a
tier stating what backs it is describing *itself*, and importing the change
filter to do so made the two tiers depend on an indexing helper — the wrong way
round, and the reason this moved.
"""


@runtime_checkable
class VersionedProvider(Protocol):
    """A provider that knows which version control backs the tree it hands out.

    An optional capability as its own Protocol — `ReclaimingProvider`'s shape and
    its reason. **`Workspace.kind` cannot answer this**: the git and jj tiers both
    hand back `worktree`, so the provider is the only thing that can say. A tier
    declaring nothing falls through to `ph.seams.changes`'s filesystem probe,
    which is the `shared` case and therefore the common one.
    """

    vcs: Backend


@runtime_checkable
class SnapshottingProvider(Protocol):
    """A tier whose version control commits the tree in order to *read* it.

    jj is the one that does: it snapshots the working copy on any command, which
    is what makes `ph.seams.changes`'s jj backend a single call — and, in the same
    breath, means a read adopts whatever is untracked. `auto_track` exists to keep
    the seam's own provisioned materials (E14) out of that commit, and it can only
    be applied by the tier, which is the only thing that knows what it provisioned.

    So a caller that needs the snapshot asks for it **here** rather than reaching
    for the raw `jj` helper, whose own docstring sends callers through the
    provider for exactly this reason. A tier that does not declare this capability
    is one whose reads cost the tree nothing.
    """

    async def snapshotting(self, cwd: Path, *args: str) -> tuple[int, str, str]: ...


@dataclass(frozen=True, slots=True)
class SharedWorkspaceProvider:
    """`workspace-shared` — today's behaviour, and the floor under every tier.

    Returns `base` itself: mounting the seam changes nothing, no checkout, no copy,
    no cost. It is also the fallback for a provider that declines, which is why it
    lives beside the seam rather than in a row of its own — "there is always a
    workspace" cannot be a promise kept by a row a profile might not layer.

    `access="read"` is honoured by *saying so*: the kind stays `shared` and
    `repo_writable` stays `True`, because nothing here enforces anything.
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

    Paired here rather than on `Workspace` because the disposer is the *scope's*,
    not the provider's: a value handed to a caller should not carry the seam's
    bookkeeping.
    """

    workspace: Workspace
    dispose: Disposer | None = None
    session: Session | None = None
    """Where this workspace's closing event goes, set once the acquisition is logged.
    The release closure reads it here rather than capturing it, so the two halves of
    the pair cannot disagree about which log they belong to.
    """


@dataclass(slots=True)
class WorkspaceSeam:
    """The service published as `ctx.workspace`."""

    ctx: Context
    shared: SharedWorkspaceProvider
    scratch_root: Path
    provider: WorkspaceProvider | None = None
    provider_by: Running | None = None
    """Who registered the tier (P6-29). Entered around every call into the
    provider; see `CompactionSeam.engine_by` for why the layer stays the
    registration's."""
    _provisioning: list[ProvisionEntry] = field(default_factory=list)
    """Materials to put in a fresh workspace, in registration order (E14).

    A list rather than a `claim_slot`, because two sources legitimately compose: the
    profile's own row, and a repository's `.ph-workspace.yml`. Later entries win a
    collision.
    """
    _held: dict[str, _Held] = field(default_factory=dict)
    """Live workspaces by agent id — keyed by id because the question is asked by
    things that have an agent id and no agent object (the prompt's workspace line,
    `ph doctor`'s per-agent report). Emptied by the effect disposer, so an entry
    surviving its agent is the same leak `workspace/acquired` without a `disposed`
    records durably.
    """

    def of(self, agent_id: str) -> Workspace | None:
        """The workspace this agent holds, if it has acquired one. `None` is a real answer
        and the common one: nothing acquires until the agent lifecycle does (P4-08).
        """
        held = self._held.get(agent_id)
        return None if held is None else held.workspace

    def provision(
        self, entries: Sequence[ProvisionEntry], *, scope: Context | None = None
    ) -> Disposer:
        """Contribute materials for every fresh workspace this seam hands out.

        On the *seam* rather than on the tier, so `readonly-scratch` (P6-05) and any
        later fresh-root kind inherit the guards without re-implementing them — the same
        argument that put `scratch` here. Nothing is provisioned into a `shared`
        workspace, whose root *is* the base.

        `scope=` registers on *someone else's* lifetime, which is all it now means
        (P6-12, P6-25): a registration made from a row's `apply`, or from a listener
        that row wrote, already unwinds with the row.

        Through `claim_entry` because a `ProvisionEntry` is a **value**: two rows
        contributing `{source: .env}` compare equal, and `list.remove` would have one
        row's disposer take the other's.
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
        """Every workspace an agent currently holds — what `/workspaces` needs before it
        offers to delete a directory.

        Matched by root rather than by inverting a directory name back into an agent id:
        `sanitize_ref` is lossy, so an id that does not sanitize to itself would read as
        unheld and lose the refusal that protects it.
        """
        return [held.workspace for held in self._held.values()]

    def register_provider(
        self, provider: WorkspaceProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim the tier. One at a time; `shared` remains the fallback."""
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "provider",
            provider,
            label="workspace.provider",
        )

    def effective_tier(self, *, child: bool) -> ContainmentTier:
        """What one role actually gets, provider and choice reconciled.

        **Effective, not configured, in both directions**: a `worktree` row over a
        directory that is not a repository declines on every acquire, and the shipped
        `rlm` profile layers the git provider while choosing `advisory` for the person's
        own agent. Reading either half alone names containment somebody does not have.

        The two halves can each only *lower* the answer and neither can raise it.
        `acquire` makes the same reconciliation by *doing* it, which is why this is the
        only other place allowed to state it.
        """
        if self.provider is None:
            return "advisory"
        containment = self.ctx.get(CONTAINMENT)
        chosen = None if containment is None else containment.for_role(child=child)
        if chosen == "advisory":
            return "advisory"
        return self.provider.tier

    def describe(self) -> list[tuple[str, str]]:
        """What `ph doctor` prints about workspaces (E10).

        **Per agent, not per profile**: since P4-11 there is no single answer — the
        shipped `rlm` posture puts the person's own agent in their checkout and its
        children in worktrees. An agent that has acquired nothing prints nothing, rather
        than inventing a row per configured agent for workspaces nobody holds.
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
        """How many trees are being kept as evidence, across stored sessions (P6-28).

        The row that makes the pile visible. The per-agent rows above cannot show it —
        `doctor` mounts a profile with no agents, so every retained tree is by
        definition one nobody holds any more.

        Printed **even when the answer is none**, on rule 6: the assumption a reader
        makes in the absence of a row is that nothing is accumulating, which is the
        assumption this row exists to check.

        Bounded by whatever `stored()` lists rather than by walking every log ever
        written, so it is a **floor, not a census**, and it says so. Broad `except` for
        `doctor`'s own reason: a profile that cannot answer one question must still
        answer the rest.
        """
        store = self.ctx.get(SESSION_PERSISTENCE)
        if store is None:
            return "unknown — no session store is mounted"
        survivors, touched = stored_survivors(store)
        found = [record for record in survivors if record.outcome == "retained"]
        if not found:
            return f"none, across the {len(touched)} most recent session(s)"
        sessions = len({record.session_id for record in found})
        # Two lines: this renders into a two-column table, and a sentence that
        # wraps at 80 columns reads as a stray.
        return (
            f"{len(found)} across {sessions} of the {len(touched)} most recent session(s)\n"
            "collect them with `ph workspaces gc`"
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
        """Take a workspace for one agent. Never fails, never returns `None`.

        **The rung is derived, not asked of the caller** (P4-11): the role is already in
        hand, because a child's session says so (`origin: "subagent"`), so no caller can
        forget it and get the provider where the shipped profile says a *root* agent
        should have to ask for the escalation.

        `tier` overrides that derivation for a caller who means something specific,
        exactly as `cwd` overrides `shell.run`'s. `advisory` declines a registered
        provider; anything else consults it.

        `scope` bounds the workspace's life — the agent's own scope. Disposing it
        releases the workspace and writes the closing event, so an error path that never
        reaches an explicit `dispose` is not a leak. **Omitted, it is still the agent's**
        when `ctx.agents` knows `agent_id` (P4-16): the caller has already said whose
        workspace this is, and the only lifetime that can own a live agent's checkout is
        that agent's. An id the registry has never seen keeps `owner_for`'s fallback.

        **Every root this hands out is canonical** (`ph.paths.canonical`), because the
        inputs are: `base` is resolved here, `scratch` is resolved in `_scratch_for`,
        and a provider's own root comes from `default_home_path`. The
        kernel matches resolved paths, and the same set is what `permissions-fs`
        prompts about — one spelling at the source is what keeps those two boundaries
        one boundary (E6, `writable_roots`).
        """
        base = canonical(base)
        scratch = await self._scratch_for(session_id, agent_id)
        chosen = self._chosen_tier(session) if tier is None else tier
        workspace = None
        declined: DeclineReason | None = None
        if self.provider is not None and chosen != "advisory":
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
                # Not an error path: half the directories a person runs pH in
                # are not repositories.
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
                # An operator's fact as much as the log's (E1): an agent that
                # was meant to be contained and is not.
                await ops_record(
                    self.ctx,
                    "workspace provider failed; the agent is not contained",
                    severity="error",
                    agent_id=agent_id,
                    base=str(base),
                )
            else:
                if workspace is None:
                    # No reason is fabricated: a provider that declined without
                    # giving one has not told us why.
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
        # Owned *before* provisioning: materialising a dependency directory is
        # thousands of syscalls, and running it before the `ctx.effect`
        # registration would leave the worktree existing with nothing to unwind
        # it — against I2, in the module that argues I2.
        if scope is None:
            scope = self._agent_scope(agent_id)
        held = await self._track(workspace, agent_id, scope)
        held.workspace = await self._provision(workspace, base)
        self._log(held.workspace, agent_id, session, declined)
        return held.workspace

    async def _provision(self, workspace: Workspace, base: Path) -> Workspace:
        """Put the configured materials in a *fresh* root (E14)."""
        if not self._provisioning or not fresh_root(workspace.kind):
            return workspace
        # Qualified: `provision` on this class is the *registration*; the module
        # function is the work.
        report: ProvisionReport = await workspace_provision.provision(
            self._provisioning, base=base, root=workspace.root
        )
        if report.failed:
            log.warning(
                "ph.seams.workspace: %d material(s) did not reach %s",
                len(report.failed),
                workspace.root,
            )
        return replace(workspace, provisioned=report.provisioned, provision_failures=report.failed)

    async def dispose(self, agent_id: str) -> None:
        """Release this agent's workspace early.

        "Early" because the scope owns it either way (I2); this is the same teardown
        reached deliberately rather than by unwinding. Calling it twice is a no-op — the
        disposer deregisters itself.
        """
        held = self._held.get(agent_id)
        if held is None or held.dispose is None:
            return
        await maybe_await(held.dispose())

    async def _scratch_for(self, session_id: str, agent_id: str) -> Path:
        """Per session *and* per agent, created rather than merely named. Owned by the seam
        so the layout has one implementation: two children of one session writing notes
        into one directory is the collision this avoids.

        Canonical here rather than trusting `scratch_root` to be: the shipped row gets
        it from `default_home_path`, which already is, but the guarantee `acquire`
        makes is the seam's, so it does not depend on how the seam was built.
        """
        scratch = canonical(self.scratch_root / session_id / agent_id)
        await anyio.to_thread.run_sync(lambda: scratch.mkdir(parents=True, exist_ok=True))
        return scratch

    def _chosen_tier(self, session: Session | None) -> ContainmentTier | None:
        """Which rung this acquisition gets, read off the deployment's choice.

        The role comes from the session rather than from an argument: a child's header
        carries `origin: "subagent"`, so "is this a child" is a fact the seam holds.

        `None` — no containment row — means nobody chose, and a deployment that layered
        a provider and never mentioned containment gets that provider: layering it *was*
        the choice.
        """
        containment = self.ctx.get(CONTAINMENT)
        if containment is None:
            return None
        child = session is not None and session.header.origin == "subagent"
        chosen: ContainmentTier | None = containment.for_role(child=child)
        return chosen

    def _agent_scope(self, agent_id: str) -> Context | None:
        """The live agent's own scope for `agent_id`, or `None` when nobody knows it.

        `acquire` with no `scope=` used to fall straight through to `owner_for(None)`,
        which outside a row's `apply` is the seam itself — a worktree taken for a real
        agent then outlived that agent by the whole process, and the containment-ladder
        tests had to remember `scope=agent.ctx` by hand (P4-16). The registry already
        knows the answer, so the seam asks it. A disposed scope is declined for
        `owner_for`'s reason: a registration on a dead lifetime is one nothing unwinds.
        """
        agents = self.ctx.get(AGENTS)
        agent = agents.get(agent_id) if agents is not None else None
        return agent.ctx if agent is not None and agent.ctx.active else None

    async def _track(self, workspace: Workspace, agent_id: str, scope: Context | None) -> _Held:
        """Register the teardown as an effect, so the workspace has an owner.

        The release closure reads `held.workspace` rather than capturing one, because
        provisioning replaces the value a moment later and the teardown policy needs the
        *final* one — what was put in the tree is what it must not mistake for the
        agent's work.
        """
        held = _Held(workspace=workspace)
        self._held[agent_id] = held

        def enter() -> Disposer:
            async def release() -> None:
                # Identity, not presence: an agent that re-acquired must not have
                # its live workspace evicted by the previous handle's disposal.
                if self._held.get(agent_id) is not held:
                    return
                del self._held[agent_id]
                current = held.workspace
                kept = True if current.release is None else await current.release(current)
                if held.session is not None:
                    held.session.append(DISPOSED, self._payload(current, agent_id, kept=kept))

            return release

        held.dispose = await self.ctx.owner_for(scope).effect(enter, label=f"workspace({agent_id})")
        return held

    def _log(
        self,
        workspace: Workspace,
        agent_id: str,
        session: Session | None,
        declined: DeclineReason | None,
    ) -> None:
        """Both halves of the durable pair are written by the seam: a pair only reconciles
        if one place owns both, and a provider that forgot the second would leave every
        workspace looking leaked.
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
            # Only when a tier was asked and could not serve: absent means
            # "no tier configured", which is a different fact and the one
            # `ph doctor` must not confuse it with (E15).
            data["declined"] = declined
        session.append(ACQUIRED, data)
        if workspace.provision_failures:
            session.append(
                "workspace/provisioned",
                {"agentId": agent_id, "failed": list(workspace.provision_failures)},
            )

    def retain(self, agent_id: str, reason: str) -> bool:
        """Keep this agent's tree past disposal, and say why (P6-28).

        Called by whoever learns how an agent ended, at any point before the scope
        unwinds; the teardown policy reads `Workspace.retained` and skips the discard.

        **An empty `reason` clears the mark**, and it is the same call because it is the
        same decision revisited: the shipped policy retains *by default* for the kind
        that discards, so a clean settle is a caller saying "never mind" — and that has
        to be as durable as the mark it withdraws.

        Returns whether anything was marked; `False` for an agent holding no workspace,
        so a caller that retains speculatively needs no `hasattr` probe and no
        exception.
        """
        held = self._held.get(agent_id)
        if held is None:
            return False
        held.workspace = replace(held.workspace, retained=reason)
        if held.session is not None:
            # Durable at the moment of marking, not only on the closing half:
            # the worst way for a run to go wrong writes no `disposed`.
            held.session.append(
                RETAINED, pair_payload(agent_id, held.workspace.ref, retained=reason)
            )
        return True

    def _payload(
        self,
        workspace: Workspace,
        agent_id: str,
        **extra: object,
    ) -> dict[str, Any]:
        payload = pair_payload(agent_id, workspace.ref, **extra)
        # The reason rides the closing half, because that is the half a fold
        # reads to tell a deliberate keep from a dirty-tree keep (P6-28).
        if workspace.retained:
            payload["retained"] = workspace.retained
        return payload

    async def reconcile(self, session: Session) -> None:
        """Close the pairs a crash left open in this session's log (F6).

        On the seam because both facts it needs are here: `_held` answers "is this tree
        anybody's", and `_payload` owns the shape of the pair, so the `disposed` a
        reconciliation writes is the one an orderly release would have written.

        A leak this profile cannot reclaim is **reported and left alone**: the tree
        belongs to a tier that is not mounted here, and removing a directory on the
        strength of a record written by a configuration we are not running is the one
        way this could destroy the work it exists to protect.
        """
        leaks = [one for one in workspace_leaks(session) if one.agent_id not in self._held]
        provider = self._reclaimer(leaks, "reclaim")
        if provider is None:
            return

        async def reclaim(record: WorkspaceRecord) -> None:
            kept = await self._reclaim(provider, record, "reclaim")
            if kept is None:
                return
            # The pair closes either way: a leak left open is one reported at
            # every future open.
            session.append(
                DISPOSED, pair_payload(record.agent_id, record.ref, kept=kept, reconciled=True)
            )

        # Concurrent: several subprocesses per leaked tree.
        async with anyio.create_task_group() as group:
            for record in leaks:
                group.start_soon(reclaim, record)

    def collectable(
        self,
        survivors: Iterable[WorkspaceRecord],
        *,
        older_than: float,
        now: float,
        touched: Mapping[str, float],
    ) -> list[Collectable]:
        """Which retained trees may be removed, and why the others may not.

        **Only `retained` trees, and that boundary is the whole safety argument.** A
        `kept` tree is a dirty checkout the disposal policy left for a person to inspect
        — `/workspaces remove` is the deliberate way to end that. A `leaked` tree
        belongs to `reconcile`, the only thing that can tell "the process died" from
        "the process is running". What a policy retained without anybody asking is all
        this may collect.

        Three refusals, in the order they are cheap to test:

        * `open` — an unclosed pair is a live process or a crash, and either way not
          this mechanism's to settle. Listed in `survivors` so an enumeration can show
          it; never collected.
        * `held` — this process holds the tree, matched by **root path** for `live()`'s
          reason: `sanitize_ref` is lossy, so an id that does not sanitize to itself
          would read as unheld and lose the refusal that protects it.
        * `recent` — inside the age bound, dated from `touched`. A session id *absent*
          from `touched` is refused as `recent` rather than collected: "I could not date
          this" and "this is old" are different answers and only one may delete a
          checkout.

        The age is the **log's** last write, not the tree's own mtime — a person reading
        a retained tree without editing it bumps neither, so no clock here detects
        interest.
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
        """Remove the trees `collectable` cleared, and answer with what went.

        **Retention is revoked, not overridden**: this hands the record back to
        `reclaim` with its reason cleared, so what runs is the disposal policy that
        would have run at release time had nobody retained the tree. Nothing here can
        destroy more than an ordinary disposal would have, which is what lets the age
        bound be a *default* rather than a decision.

        Sequential, unlike `reconcile`'s fan-out: this is a person watching a command
        they typed, where a failure halfway through a fan-out is a report they cannot
        act on.

        Nothing is appended — the pair is already closed, and `reclaim` answers `False`
        for a directory that is not there, so re-running is a no-op either way.
        """
        wanted = [row.record for row in rows if row.verdict == "collect"]
        provider = self._reclaimer(wanted, "collect")
        if provider is None:
            return []
        removed: list[WorkspaceRecord] = []
        for record in wanted:
            # Revoked, not overridden: `reason=""` is what makes `reclaim` run
            # the disposal policy it would have run had nobody retained the tree.
            if await self._reclaim(provider, replace(record, reason=""), "collect") is False:
                removed.append(record)
        return removed

    async def export(self, record: WorkspaceRecord) -> str | None:
        """The ref this agent's work is on, or `None` if no tier can say.

        `None` rather than a raise, matching `_reclaimer`: a profile whose tier cannot
        export is not a broken deployment, it is one where the answer is "there is
        nothing to move".
        """
        provider = self.provider
        if not isinstance(provider, ExportingProvider):
            log.warning("ph.seams.workspace: no mounted tier can export %s", record.agent_id)
            return None
        return await provider.export(record)

    def _artifacts(self) -> ArtifactProvider | None:
        """The mounted tier's ref verbs, or `None` — `_reclaimer`'s shape, one line."""
        provider = self.provider
        return provider if isinstance(provider, ArtifactProvider) else None

    def _checkouts(self) -> EnumeratingProvider | None:
        provider = self.provider
        return provider if isinstance(provider, EnumeratingProvider) else None

    def _checkpointer(self, workspace: Workspace) -> CheckpointingProvider | None:
        """The tier able to checkpoint *this* workspace, or `None` (P6-20).

        Both halves of `can_checkpoint` in one place, so the predicate and the
        narrowing cannot come apart — it was the same `isinstance` written twice,
        once to decide and once to convince the type checker.

        `fresh_root` is the **kind** half: a `shared` workspace's root is the
        person's own checkout, and offering to overwrite their uncommitted work with
        whatever an agent found is the one thing this must never do. The Protocol
        test is the **tier** half, which replaced a kind-keyed table of provider
        facts kept where no provider could see it.
        """
        provider = self.provider
        if not fresh_root(workspace.kind) or not isinstance(provider, CheckpointingProvider):
            return None
        return provider

    async def refs(self, base: Path) -> list[str]:
        """Every ref in this repository, as the mounted tier lists them.

        Unfiltered on purpose: `/workspaces` applies `BRANCH_PREFIX` itself, because
        that prefix is the whole of what keeps it from offering to delete a person's
        own branch, and `merge` deliberately accepts a ref outside it.
        """
        provider = self._artifacts()
        if provider is None:
            return []
        with running(self.provider_by):
            return await provider.refs(base)

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        """Delete one ref. `""` when it went, else the tier's own reason.

        `force` is the caller's decision, not the tier's: both tiers refuse a ref
        holding work nothing else has, and a person overriding that is answering a
        question only they can.
        """
        provider = self._artifacts()
        if provider is None:
            return f"no mounted tier can delete {ref}"
        with running(self.provider_by):
            return await provider.delete_ref(base, ref, force=force)

    async def merge(self, base: Path, ref: str) -> str:
        """Merge `ref` where the person is standing. `""` when it merged cleanly.

        Anything else is the tier's sentence, which is not always a failure: jj
        records conflicts *in the commit* and exits zero, so "merged, with conflicts
        at these paths" is a true thing this can answer and a bool could not.
        """
        provider = self._artifacts()
        if provider is None:
            return f"no mounted tier can merge {ref}"
        with running(self.provider_by):
            return await provider.merge(base, ref)

    async def strays(self, base: Path, *, with_status: bool = True) -> dict[str, Stray]:
        """Checkouts the mounted tier still has on disk, by ref. `{}` if none can say.

        Keyed by ref rather than returned as a list, because every caller is joining
        it against something — `/workspaces` against the branches pH made — and a
        list would have each of them build the same index.
        """
        provider = self._checkouts()
        if provider is None:
            return {}
        # The refs live agents hold, so no tier spends a probe on a `dirty` the
        # caller discards — `describe` reports a held row as held and reads no
        # further. The seam supplies it because `live()` is the seam's.
        held = frozenset(workspace.ref for workspace in self.live() if workspace.ref)
        with running(self.provider_by):
            found = await provider.strays(base, with_status=with_status, skip=held)
        return {one.ref: one for one in found}

    async def discard(self, path: Path) -> str:
        """Remove one checkout. `""` when it went, else the sentence saying why not.

        A message rather than a bool, because the only caller is a person who typed
        `remove` and is owed the tier's own reason. A profile whose tier cannot
        enumerate cannot have produced this path in the first place, so the refusal
        is a guard rather than a case.
        """
        provider = self._checkouts()
        if provider is None:
            return f"no mounted tier can remove {path}"
        with running(self.provider_by):
            return await provider.discard(path)

    def can_checkpoint(self, workspace: Workspace) -> bool:
        """Whether a restore point can be taken for this workspace at all (P6-20).

        The gate `/revert` asks before offering one, so a workspace that can never
        have a restore point **refuses** rather than reporting "no restore points in
        this session" — true, useless, and indistinguishable from a run that had
        simply not checkpointed yet.

        Both halves live in `_checkpointer`, which is also what `capture` narrows
        through — the predicate and the narrowing were the same `isinstance` written
        twice, and two spellings of one gate is one that can come apart.
        """
        return self._checkpointer(workspace) is not None

    async def capture(self, workspace: Workspace) -> str | None:
        """A token naming this workspace's state now, or `None` if no tier can say.

        `None` rather than a raise, matching `export`: a profile whose tier cannot
        checkpoint is not a broken deployment, and the fingerprint's consumer reads
        an empty answer as "always re-run", which is the safe direction.
        """
        provider = self._checkpointer(workspace)
        if provider is None:
            return None
        with running(self.provider_by):
            return await provider.capture(workspace)

    async def checkpoint(
        self, workspace: Workspace, *, session: Session, agent_id: str, call_id: str
    ) -> str | None:
        """Capture a restore point and record it. `None` when the tier has none to give.

        The event is the *seam's*, for `_log`'s reason: two tiers write restore
        points and three consumers fold them, so the payload's shape belongs to
        neither tier.

        **Pinned before it is recorded**, which reverses the order the git tier used
        while it owned this. That ordering existed for a mechanical reason — the ref
        was named by the event's own `seq`, so a pin written first had no name to
        write — and a token now names its own pin. What that changes is the
        direction a crash between the two can fail in, and the new one is better: it
        loses the *record* of a restore point whose state is safely pinned, where
        before it kept a record of state that had not been pinned yet. A pin nobody
        recorded is idempotent garbage the next capture writes over.

        `tree` is the payload key, and it is historical rather than descriptive: the
        git tier's token is a tree, jj's is a commit. Renaming it would break every
        reader of a log written before today for no gain a person can see.
        """
        token = await self.capture(workspace)
        if token is None:
            return None
        session.append(CHECKPOINT, {"agentId": agent_id, "tree": token, "callId": call_id})
        return token

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]:
        """Put this workspace back to `token`. Returns the paths the run had added.

        Raises rather than answering `None`, unlike every other optional capability
        here: each caller is acting on a restore point a person or a retry ladder
        asked for **by name**, and "it silently did nothing" is the one answer none
        of them may mistake for success.
        """
        provider = self.provider
        if not isinstance(provider, CheckpointingProvider):
            raise FileNotFoundError(f"no mounted tier can restore {workspace.root}")
        with running(self.provider_by):
            return await provider.restore(workspace, token)

    def _reclaimer(
        self, records: Sequence[WorkspaceRecord], verb: str
    ) -> ReclaimingProvider | None:
        """The mounted tier, or `None` having said which trees nobody can end.

        Once per batch rather than once per record: a profile with no reclaiming tier
        owes one sentence naming what it cannot touch, not one per directory. The
        refusal is the point — removing a directory on the strength of a record written
        by a configuration we are not running is the one way either caller could destroy
        the work it exists to protect.
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

    async def _reclaim(
        self, provider: ReclaimingProvider, record: WorkspaceRecord, verb: str
    ) -> bool | None:
        """One call into a tier's teardown: whether it **kept**, or `None` if it raised.

        A tri-state answer rather than an exception, because neither caller may abort
        its batch: `reconcile` would leave the rest of a crash's pairs open, and
        `collect` would stop at the first tree git is unhappy about.

        `verb` reaches the log only, telling an operator whether a warning came from a
        reconciliation at session open or from a `gc` they typed.
        """
        try:
            with running(self.provider_by):
                return await provider.reclaim(record)
        except Exception:
            log.warning("ph.seams.workspace: could not %s %s", verb, record.root, exc_info=True)
            return None


WorkspaceOutcome: TypeAlias = Literal["leaked", "kept", "retained"]
"""Why a tree is still on disk — the three things `git worktree list` reports
identically (P6-28).

`leaked` is an `acquired` the log never saw closed, the only one nobody decided.
`kept` is the disposal policy keeping a dirty tree for review. `retained` is
somebody naming a reason, and it **wins over the other two**: a tree that was
asked for is asked for whether or not the process that held it exited cleanly.
"""


@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    """One tree a session left on disk, and why.

    What survives a crash, and only that: the log's own fields. Not a `Workspace` —
    `scratch`, `env` and the `release` closure are process state that died with the
    process, and a value carrying empty versions of them would invite a caller to
    use them.
    """

    agent_id: str
    kind: WorkspaceKind
    root: Path
    ref: str | None = None
    reason: str = ""
    """Why it was retained; empty for the other two outcomes."""
    closed: bool = False
    """Whether the durable pair reconciled — a separate axis from `reason`.

    A retention is marked *while the agent is live*, so a process that then dies
    leaves a record that is both retained and unclosed. One axis would force a
    choice between two wrong answers: leaked, and reconciliation discards the
    evidence it was told to keep; retained, and the pair never closes, so the tree
    is re-reported at every open forever.
    """
    session_id: str = ""
    """Whose log this came from. Not derivable from `agent_id` — a cross-session reader
    (the family fold, the collector) needs to get back to the log, and inferring it
    from a sanitised directory name is the lossy round-trip `/workspaces` refuses to
    make.
    """

    @property
    def outcome(self) -> WorkspaceOutcome:
        """Which of the three this is — **derived from the two facts, never stored**.

        A third name for what `reason` and `closed` already say is a third chance to
        disagree with them: `collect` revokes a retention by clearing `reason`, and a
        stored field went on reporting `retained` after the decision was withdrawn.
        """
        if self.reason:
            return "retained"
        return "kept" if self.closed else "leaked"


CollectVerdict: TypeAlias = Literal["collect", "held", "recent", "gone"]
"""What the collector decided about one retained tree. Every one is a sentence a
person reads: `held` and `recent` are refusals they may want to argue with,
`gone` is a tree somebody already removed by hand.
"""


@dataclass(frozen=True, slots=True)
class Collectable:
    """One retained tree, with the collector's verdict and how old it is.

    A verdict per record rather than a filtered list, because the refusals are the
    useful half: "nothing to collect" and "three trees, all still held by live
    sessions" are very different answers to `ph workspaces gc`.
    """

    record: WorkspaceRecord
    verdict: CollectVerdict
    age: float
    """Seconds since the owning session's log was last written."""


def workspace_survivors(session: Session) -> list[WorkspaceRecord]:
    """Every tree this session left on disk, and which of three reasons put it there.

    **The log is the only thing that can tell the three apart** — they all leave a
    worktree that `git worktree list` reports identically, so `/workspaces` cannot.
    Two are features and the third is the leak F6 exists to close.

    Folded rather than tracked, for `subagent_roster`'s reason one seam over: the
    answer has to be computable from a log being read off disk by a process that was
    not running when it was written.

    **From `seed_length`, not from the beginning.** A fork seeds the child with the
    parent's transcript, so a fold over the whole log reports the parent's
    still-held worktrees as the child's — and reconciliation would then remove a
    tree an agent is actively working in. What a session inherited is not what it
    acquired.

    Only kinds with a fresh root can leave a directory behind: a `shared`
    workspace's root *is* the base, so an unclosed pair there records a crash and no
    stray.
    """
    open_records: dict[str, WorkspaceRecord] = {}
    closed: list[WorkspaceRecord] = []
    for event in session.events[session.header.seed_length or 0 :]:
        # The type test first: a long log is mostly `assistant/chunk`, and
        # reading `agentId` off every one of them to discover it is absent costs
        # a mapping get and a string per event.
        if event.type not in _SURVIVOR_TYPES:
            continue
        data = event.data
        agent_id = as_str(data.get("agentId"))
        if not agent_id:
            continue
        if event.type == RETAINED:
            # Marked while the agent was live, so the record is already open —
            # unless this session only ever *seeded* the acquire, in which case
            # there is nothing here to mark and nothing on disk we may claim.
            marked = open_records.get(agent_id)
            if marked is not None:
                # An empty reason is a withdrawal, and it returns the record to
                # the outcome an open acquire has by default. Reading it as a
                # retention with a blank reason would make a clean settle the
                # thing that pins a tree forever.
                open_records[agent_id] = replace(marked, reason=as_str(data.get("retained")))
            continue
        if event.type == DISPOSED:
            record = open_records.pop(agent_id, None)
            if record is None:
                continue
            # A reason wins over `kept`, and over a `kept: false` too: the
            # closing half repeats it precisely so an orderly release says it,
            # and a policy that discarded a tree it had been asked to keep would
            # have written `kept: false` about a directory that is still there.
            reason = as_str(data.get("retained") or record.reason)
            if reason or data.get("kept"):
                closed.append(replace(record, closed=True, reason=reason))
            continue
        # A Literal read off JSON is a claim to check, not a cast to make.
        # Built once at import beside the alias rather than per event:
        # `get_args` is not memoized and rebuilds its tuple on every call.
        kind = _WORKSPACE_KINDS.get(as_str(data.get("kind")), "shared")
        if not fresh_root(kind):
            continue
        ref = data.get("ref")
        open_records[agent_id] = WorkspaceRecord(
            agent_id=agent_id,
            kind=kind,
            root=Path(as_str(data.get("root"))),
            ref=str(ref) if ref else None,
            session_id=session.id,
        )
    return [*open_records.values(), *closed]


def workspace_leaks(session: Session) -> list[WorkspaceRecord]:
    """Workspaces this session took and never released (F6).

    A filter over `workspace_survivors`, not a fold of its own: the two questions
    differ by one predicate and share every rule that is easy to get wrong — the
    seed offset, the fresh-root test, which acquire is the live one — and here a
    second implementation would disagree about whether a directory may be deleted
    (A11).

    The predicate is **`closed`, not `outcome == "leaked"`**: a tree somebody
    retained before the process died is a survivor with a reason *and* an open pair.
    Reconciliation still owes it the closing event, and it is `reclaim`'s job to
    know that a reason means leave the directory alone.
    """
    return [one for one in workspace_survivors(session) if not one.closed]


def checkpoints(session: Session) -> dict[int, dict[str, Any]]:
    """Every restore point in this session, by the event's own seq — a fold.

    A checkpoint is a fact in the log, so a resumed or forked session finds the same
    restore points a live one has, without anything having remembered them.
    """
    return {event.seq: dict(event.data) for event in session.events if event.type == CHECKPOINT}


@dataclass(frozen=True, slots=True)
class _CheckpointOf:
    """One agent's restore point, read off a `workspace/checkpoint` event.

    Frozen, so it hashes by value — which is what gets each agent its own fold
    out of one parser class. `Session.projection` keys on `(event_type, parse)`,
    so `_CheckpointOf("a")` and `_CheckpointOf("b")` are two keys and a closure
    (a new object per call, hashing by identity) would be a new fold per call.
    """

    agent_id: str

    def __call__(self, event: SessionEvent) -> str | None:
        """The tree, or `None` when some other agent took this checkpoint.

        `""` is a value here, not a miss: a checkpoint that recorded no tree is
        still the newest one this agent took, and answering with an *older* tree
        would revert further than the log says to.
        """
        if as_str(event.data.get("agentId")) != self.agent_id:
            return None
        return as_str(event.data.get("tree"))


def latest_checkpoint(session: Session, agent_id: str) -> str:
    """The newest restore point *this agent* took, or `""` if it has none.

    An incremental fold rather than `checkpoints()` plus `max()`: the caller that
    wants one restore point does not need a dict of every restore point, and
    building it copies each payload to discard all but the last — on a crash path
    that runs once per retry. It is not a reverse scan of `session.events`
    either, which materialised a snapshot of the whole log to read one field;
    `Session.projection` keeps the fold and parses only what has arrived since.

    Scoped to the agent, which is the rule `/revert` already states: a restore point
    belongs to the agent that took it. Only one agent writes into a root session
    today, so this is a latent difference rather than a live one — it is here so the
    two readers of this fold cannot disagree about it later. The agent rides in
    the *parser*, which is half `Session.projection`'s key — so two agents get
    two folds without this having to invent a name for either.
    """
    return session.projection(CHECKPOINT, _CheckpointOf(agent_id)) or ""


def stored_survivors(
    store: Any,  # noqa: ANN401
    *,
    limit: int = 50,
    family: str = "",
) -> tuple[list[WorkspaceRecord], dict[str, float]]:
    """Every tree the *store* can still account for, and when each log was written.

    The deployment-wide half of the fold, where `family_survivors` is the per-parent
    one. Both consumers — `ph doctor`'s count and the collector — need the same two
    things, and a second loop over `stored()` is where a listing limit and a
    tolerance rule quietly diverge.

    **A session that will not read is skipped, not fatal**, and the direction is the
    safe one for both consumers: doctor under-counts and the collector removes
    nothing, which is what you want from a half-written file. Logged, because a
    store that cannot read most of what it listed is a real problem wearing a small
    number. Since reference-forking there are two ways a log will not read, and a
    *good* file whose ancestor was removed takes every descendant with it — so this
    count can fall by more than the number of damaged files. `ph doctor`'s "Session
    lineage" section is what answers which.

    `family` narrows the answer to one agent and its descendants, through
    `descendants` so this and `family_survivors` cannot disagree about who counts.
    Applied to the **listing**, before anything is read: `StoredSession` already
    carries `parent`, and descent needs nothing else.

    `touched` is **not** narrowed with it — the caller is reporting how much of the
    store it looked at, and a denominator that shrank with the filter would say a
    family's trees came from every session on disk.

    **Folded and discarded one at a time.** A stored log is seeded through the same
    surface validation a resume makes, which is what stops this counting a tree in a
    log the harness would refuse to reopen; the fold reads three event types and
    keeps nothing else.

    `limit` is whatever `stored()` will show, which makes the answer a **floor**
    rather than a census — both callers say so in their own words.
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
        else set(descendants(((one.session_id, one.parent) for one in listed), family))
    )
    for entry in listed:
        if entry.session_id not in wanted:
            continue
        try:
            header, events = store.read(entry.session_id)
            survivors.extend(workspace_survivors(Session(entry.session_id, events, header)))
        except Exception:
            log.warning(
                "ph.seams.workspace: could not read session %s", entry.session_id, exc_info=True
            )
    return survivors, touched


def family_survivors(sessions: Sequence[Session], agent_id: str) -> list[WorkspaceRecord]:
    """What one agent and everything beneath it left on disk (P6-28).

    **The fold a parent cannot do from its own log**, because a child's workspace
    events are in the *child's* log. The link already exists: a child's session
    names its parent in its own header, so this is a walk over state rather than an
    index to maintain.

    Ordered **parent-first, then by descent** — what the agent I asked about left,
    then what it delegated. `descendants` is breadth-first and this preserves it.

    Sessions are passed in, not fetched, for `reachable_family`'s reason: the caller
    has already decided which logs it may open. A session named by `agent_id` that
    is not in `sessions` yields nothing rather than raising, because a truncated
    listing is an ordinary answer.
    """
    by_id = {session.id: session for session in sessions}
    lineage = [(session.id, session.header.parent_session) for session in sessions]
    return [
        record
        for one in descendants(lineage, agent_id)
        if (session := by_id.get(one)) is not None
        for record in workspace_survivors(session)
    ]


def pair_payload(agent_id: str, ref: str | None, **extra: object) -> dict[str, Any]:
    """The keys both halves of the durable pair share, spelled once.

    `ref` rides both so a reader can say which branch a turn ran against without
    inspecting the repository, and is omitted rather than sent as `null` for kinds
    that have none. A module function because the pair has two writers — an orderly
    release and a reconciliation — and the second is the half nobody watches.
    """
    data: dict[str, Any] = {"agentId": agent_id, **extra}
    if ref is not None:
        data["ref"] = ref
    return data


ACQUIRED = "workspace/acquired"
DISPOSED = "workspace/disposed"
"""The durable pair, named once: the fold below and both producers have to agree on
these exactly.
"""

RETAINED = "workspace/retained"
"""A tree marked as evidence, recorded the moment it is marked (P6-28).

**Not part of the pair, and the reason is the crash.** The decision records that
a run went wrong, and the most complete way for a run to go wrong is for the
process to die — which writes no `disposed` at all. A retention held only in
memory would be lost by exactly the failure it exists to survive, and
reconciliation would discard the tree it was told to keep.

Ignorable: an older build that skips it reads a keep as an ordinary keep.
"""

CHECKPOINT = "workspace/checkpoint"
"""One restore point, recorded the moment it is taken (P4-09).

Here rather than in the git tier that first wrote it, because the capability is
the seam's now: two providers append this and three consumers fold it, so an
event name spelled in one of them would be a vocabulary the other borrowed.

Not part of the acquire/dispose pair — a checkpoint opens nothing and closes
nothing, and a fold over the pair must ignore it.
"""


_SURVIVOR_TYPES = frozenset({ACQUIRED, DISPOSED, RETAINED})

_WORKSPACE_KINDS: Mapping[str, WorkspaceKind] = literal_lookup(WorkspaceKind)
"""Every `WorkspaceKind` by its own spelling — the read-side check for a kind
named in a stored log, which a different build may have written. See
`literal_lookup`.

Built here rather than in the fold: tested once per event in the hot loop."""

PROJECT_PROVISION_FILE = ".ph-workspace.yml"
"""Where a repository states what its worktrees need (E14).

Discovered by walking up from the project directory, and read for **data only**:
a `copy`/`symlink`/`hardlink` entry, nothing that executes. That is what makes
cloning a repository and starting pH safe, and why the `command` hook was
refused rather than trust-gated.

Read once, at mount; a file that appears later needs a restart.
"""


class LifecycleConfig(WireModel):
    """Row config for the lifecycle."""

    access: WorkspaceAccess = "write"
    """What the *root* agent needs of the project directory.

    A child's access is its parent's to decide and arrives with the spawn
    (E4, Q11); this is the person at the keyboard, who asked for a harness in
    their own repository.
    """
    provision: list[ProvisionEntry] = Field(default_factory=list)
    """Materials every fresh workspace gets — `.env`, a dependency directory, a
    local config the project gitignores (E14). Empty by default: a profile that
    names none provisions none, and a `shared` workspace is never provisioned at
    all because its root already *is* the base."""


@plugin("workspace-lifecycle", inject=[WORKSPACE, FS], config=LifecycleConfig)
async def lifecycle(ctx: Context, config: LifecycleConfig) -> None:
    """Give every agent a workspace, and point `ctx.fs` at it.

    **The seam alone changes nothing; this row is what makes a tier bite.** Separate
    from the seam's own row because the two answer different questions — "what
    happens when someone acquires" and "who acquires, and when" — and a deployment
    driving the lifecycle itself wants the first without the second.

    Acquisition is *lazy and idempotent*, at the first `agent/pre-step`.
    `agent/created` is an `emit`, so a listener that has to `await git worktree add`
    could not hold the agent up and the first tool call would race the checkout. And
    a child's workspace is its parent's decision — base and `access` both — so the
    spawn path acquires first and this row must find that one rather than overwrite
    it.
    """

    def root_of(agent: AgentHandle) -> Path | None:
        workspace = workspace_of(ctx, agent)
        return None if workspace is None else workspace.root

    ctx.require(FS).rebase(root_of, scope=ctx)

    # The profile's list and the project's, composed here rather than by two
    # registrations: `provision()` accepts many contributors, and this row is
    # simply the one that knows about both sources.
    entries = [*config.provision, *discover_provisioning(ctx.require(FS).root)]
    if entries:
        ctx.require(WORKSPACE).provision(entries, scope=ctx)

    async def ensure(
        request: PreStepRequest,
        next_: Next[PreStepDecision],
    ) -> PreStepDecision:
        agent = request.agent
        if ctx.require(WORKSPACE).of(agent.id) is None:
            if request.session.header.origin == "subagent":
                # Refused rather than answered: see `ChildWorkspaceMissing`. The
                # test is the seam's own — `_chosen_tier` reads the same field to
                # decide which rung a child gets, so "is this a child" has one
                # spelling here and cannot come apart from the tier decision.
                raise ChildWorkspaceMissing(
                    f"agent {agent.id} is a subagent with no workspace: its base and access "
                    "are its parent's to decide and arrive with the spawn, so acquiring one "
                    "here would grant it more than its admission recorded"
                )
            await ctx.require(WORKSPACE).acquire(
                session_id=request.session.id,
                agent_id=agent.id,
                # The process's directory, never `fs.root_for(agent)`: that is
                # the workspace we are about to take, and branching a worktree
                # from the previous one would nest a checkout per turn.
                base=ctx.require(FS).root,
                access=config.access,
                session=request.session,
                # The agent's own scope, so the worktree is released when the
                # agent is — the in-process half of cleanup (I2), with the event
                # pair covering the crash the scope cannot (§4.9).
                scope=agent.ctx,
            )
        return await next_()

    # Outermost, so a listener that reads or writes files during the step —
    # compaction's summariser, a permissions row — sees the agent's own root
    # rather than the process's.
    ctx.on("agent/pre-step", ensure, prepend=True)


def discover_provisioning(start: Path) -> list[ProvisionEntry]:
    """The project's own materials list, walking up from `start`.

    Nearest-first and *first-wins*, `memory-agents-md`'s rule: the file beside the
    code knows what the code needs, and a monorepo root should not override a
    package that states its own.

    Read with `safe_yaml_load`, the same reader every profile row goes through —
    this is the *least* trusted config the harness opens, so it is the last one that
    should have its own parsing rules.

    **Every failure is a shrug**: a malformed file, an unknown key, a `source`
    naming somewhere outside the tree. Refusing to start because a repository's
    optional config is wrong would make this list load-bearing, and `resolve_entry`
    refuses the dangerous entries individually anyway.
    """
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_PROVISION_FILE
        if not candidate.is_file():
            continue
        try:
            document = (
                safe_yaml_load(candidate.read_text(encoding="utf-8"), origin=str(candidate)) or {}
            )
            raw = document.get("provision", []) if isinstance(document, dict) else []
            return [ProvisionEntry.model_validate(item) for item in raw]
        except Exception:
            log.warning("ph.seams.workspace: ignoring %s", candidate, exc_info=True)
            return []
    return []


class Config(WireModel):
    """Row config for the shared provider."""

    scratch: str | None = None
    """Where scratch directories live. `$PH_HOME/scratch` by default — the idiom
    `default_home_path` exists for, and outside the workspace on purpose."""


@plugin("workspace-shared", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam with the shared provider as its floor.

    Named for the provider rather than for the seam, because there is no useful
    "seam with no behaviour" state here: `sandbox-policy` can mount a seam whose
    `confine()` refuses, since refusing is a real answer, but an agent with no
    working directory is not.
    """
    seam = WorkspaceSeam(
        ctx=ctx,
        shared=SharedWorkspaceProvider(),
        scratch_root=default_home_path(config.scratch, "scratch"),
    )
    ctx.provide(WORKSPACE, seam)

    contribute(ctx, Diagnostic(id="workspaces", title="Workspaces", read=seam.describe, order=20))


@plugin("workspace-reconcile", inject=[WORKSPACE])
async def reconcile(ctx: Context, config: None) -> None:
    """Run the seam's reconciliation whenever a session is opened (F6).

    **On `session/created`, which is also the resume path** — `sessions.adopt`
    publishes through it, so a session coming off disk meets the same listener as a
    fresh one, and a fresh one folds an empty log. One mechanism rather than a
    resume-only hook that a second way of opening a session would miss.

    Detached, because `emit` schedules an async listener and does not wait:
    reconciliation runs `git` per leaked tree. `ctx.drain()` is what a test — or a
    shutdown — uses to know it has settled.
    """
    # Catch-up, for the reason `session-persistence-jsonl` does the same: a row
    # activated after sessions already exist owes them what a fresh one gets.
    for session in ctx.require(SESSIONS).list():
        await ctx.require(WORKSPACE).reconcile(session)
    ctx.on("session/created", ctx.require(WORKSPACE).reconcile)


@plugin("workspace-checkpoint", inject=[TOOLS, WORKSPACE])
async def checkpoint_policy(ctx: Context, config: None) -> None:
    """Take a restore point before every code run that has a workspace to save.

    Around the *transport*, because a run is the unit that can be denied with work
    already done (Q9a) — a native call that is denied never ran, so there is nothing
    to restore it to. The transport is identified by the view's own
    `transport_name`, since a profile may present it as `ipython`.

    **Tier-agnostic since the capability became the seam's.** This was a git row
    that cached git directories and asked a kind-keyed predicate, so a second tier
    with restore points would have needed a second copy of the policy to get them —
    and would have had to be added to that predicate to be allowed to. It now asks
    the seam, and `subprocess` is gone from `inject` because nothing here spawns
    anything any more.
    """

    async def around(
        execution: ToolExecution,
        next_: Next[ToolExecutionResult],
    ) -> ToolExecutionResult:
        # A failure here never blocks the run: a missing restore point is worse than
        # no restore point only if it is believed in, and the log records which runs
        # have one. The guard is inside the `try` on purpose — reading the tool view
        # is itself a call that must not take a cell down.
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

    ctx.on("tools/execute", around)
