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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import anyio
from pydantic import Field

from ..cordis import Context, Disposer, Running, maybe_await, plugin, running, safe_yaml_load
from ..paths import default_home_path
from ..session import Session
from ..wire import WireModel
from . import workspace_provision
from ._registry import claim_entry, claim_slot
from .diagnostics import Diagnostic, contribute
from .sandbox import SandboxPolicy
from .subagents import descendants
from .workspace_provision import ProvisionEntry, ProvisionReport

__all__ = [
    "PROJECT_PROVISION_FILE",
    "CollectVerdict",
    "Collectable",
    "ContainmentTier",
    "DeclineReason",
    "ExportingProvider",
    "LifecycleConfig",
    "ReclaimingProvider",
    "SharedWorkspaceProvider",
    "Workspace",
    "WorkspaceAccess",
    "WorkspaceDeclined",
    "WorkspaceKind",
    "WorkspaceOutcome",
    "WorkspaceProvider",
    "WorkspaceRecord",
    "WorkspaceSeam",
    "apply",
    "discards_writes",
    "discover_provisioning",
    "family_survivors",
    "fresh_root",
    "lifecycle",
    "project_access",
    "redirection_env",
    "restorable",
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


def restorable(kind: WorkspaceKind) -> bool:
    """Whether this kind has a restore mechanism at all (P6-20).

    The gate `/revert` asks before offering a restore point, so that a kind which
    can never have one **refuses** rather than reporting "no restore points in this
    session" — true, useless, and indistinguishable from a run that simply had not
    checkpointed yet.

    An overlay is `False` today rather than forever: its delta is a perfectly good
    restore point, it is simply not a git tree. Giving it one is a
    `CheckpointingProvider`, the shape `acquire` and `reclaim` already take.
    """
    match kind:
        case "worktree" | "worktree-ephemeral":
            return True
        case "shared" | "readonly-scratch" | "overlay" | "overlay-ephemeral":
            return False


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


def workspace_of(ctx: Context, agent: Any) -> Workspace | None:
    """This agent's workspace, asked of a seam that may not be mounted.

    The question written once, for the five callers that had it: the prompt line,
    `bash`, the kernel, the spawn path and the fs resolver.

    **Fail-soft on purpose.** A caller asking "where does this agent write" during a
    teardown, or in a profile that layers no workspace row, gets `None` and carries
    on with the process's own directory — a raising seam here would make an absent
    optional row fatal.
    """
    seam = ctx.get("workspace")
    if seam is None or agent is None:
        return None
    agent_id = agent if isinstance(agent, str) else getattr(agent, "id", "")
    try:
        found: Workspace | None = seam.of(agent_id)
    except Exception:
        log.warning("ph.seams.workspace: lookup failed for %s", agent_id, exc_info=True)
        return None
    return found


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

    The answer is only knowable here — P4-08's policy is "keep dirty, remove clean,
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
        return [".", *(f":(exclude){entry}" for entry in self.provisioned)]


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


@runtime_checkable
class WorkspaceProvider(Protocol):
    """A tier's implementation. `None` declines, and declining is normal.

    Typed rather than duck-typed, and `tier` is a member rather than a `getattr`
    probe: a provider whose method drifted would otherwise fail at runtime inside
    the seam's `except`, be reported to the operator as `shared`, and take the
    containment with it silently.
    """

    tier: ContainmentTier

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
        containment = self.ctx.get("containment")
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
        store = self.ctx.get("session_persistence")
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
        reaches an explicit `dispose` is not a leak.
        """
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
        """
        scratch = self.scratch_root / session_id / agent_id
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
        containment = self.ctx.get("containment")
        if containment is None:
            return None
        child = session is not None and session.header.origin == "subagent"
        chosen: ContainmentTier | None = containment.for_role(child=child)
        return chosen

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

    def _payload(self, workspace: Workspace, agent_id: str, **extra: Any) -> dict[str, Any]:
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
        agent_id = str(data.get("agentId") or "")
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
                open_records[agent_id] = replace(marked, reason=str(data.get("retained") or ""))
            continue
        if event.type == DISPOSED:
            record = open_records.pop(agent_id, None)
            if record is None:
                continue
            # A reason wins over `kept`, and over a `kept: false` too: the
            # closing half repeats it precisely so an orderly release says it,
            # and a policy that discarded a tree it had been asked to keep would
            # have written `kept: false` about a directory that is still there.
            reason = str(data.get("retained") or record.reason)
            if reason or data.get("kept"):
                closed.append(replace(record, closed=True, reason=reason))
            continue
        kind: WorkspaceKind = data.get("kind", "shared")
        if not fresh_root(kind):
            continue
        ref = data.get("ref")
        open_records[agent_id] = WorkspaceRecord(
            agent_id=agent_id,
            kind=kind,
            root=Path(str(data.get("root", ""))),
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


def stored_survivors(
    store: Any, *, limit: int = 50, family: str = ""
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


def pair_payload(agent_id: str, ref: str | None, **extra: Any) -> dict[str, Any]:
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

_SURVIVOR_TYPES = frozenset({ACQUIRED, DISPOSED, RETAINED})
"""Hoisted out of the fold: tested once per event in the hot loop."""

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


@plugin("workspace-lifecycle", inject=["workspace", "fs"], config=LifecycleConfig)
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

    def root_of(agent: Any) -> Path | None:
        workspace = workspace_of(ctx, agent)
        return None if workspace is None else workspace.root

    ctx.fs.rebase(root_of, scope=ctx)

    # The profile's list and the project's, composed here rather than by two
    # registrations: `provision()` accepts many contributors, and this row is
    # simply the one that knows about both sources.
    entries = [*config.provision, *discover_provisioning(ctx.fs.root)]
    if entries:
        ctx.workspace.provision(entries, scope=ctx)

    async def ensure(request: Any, next_: Callable[..., Any]) -> Any:
        agent = request.agent
        if ctx.workspace.of(agent.id) is None:
            await ctx.workspace.acquire(
                session_id=agent.session.id,
                agent_id=agent.id,
                # The process's directory, never `fs.root_for(agent)`: that is
                # the workspace we are about to take, and branching a worktree
                # from the previous one would nest a checkout per turn.
                base=ctx.fs.root,
                access=config.access,
                session=agent.session,
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
    ctx.provide("workspace", seam)

    contribute(ctx, Diagnostic(id="workspaces", title="Workspaces", read=seam.describe, order=20))


@plugin("workspace-reconcile", inject=["workspace"])
async def reconcile(ctx: Context, config: Any) -> None:
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
    for session in ctx.sessions.list():
        await ctx.workspace.reconcile(session)
    ctx.on("session/created", ctx.workspace.reconcile)
