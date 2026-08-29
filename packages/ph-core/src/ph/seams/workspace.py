"""`ctx.workspace` — where an agent's writes land, and how honestly that is stated.

The seam the containment ladder hangs off (D21, §4.8). Its consumer is the
**agent lifecycle**, not a tool: an agent acquires a workspace, and `ctx.fs`'s
root and `ctx.subprocess`'s cwd resolve to `workspace.root`, which is what makes
a tier bound *authored* code rather than merely observe it. A tool the model
calls could never do that job — a permission row can deny `edit`, but it cannot
deny `open(path, "w")`, because a deny-list needs a registered name to match.

**`repo_writable` is a claim, and the seam refuses to overstate it.** The
`worktree` tier gives an agent its own checkout: collisions are isolated and a
run is revertible, but an absolute-path `open()` never consults a cwd, so that
tier bounds nothing about `/etc/passwd`. Only `sandbox` can refuse that write, at
the kernel. So a caller asking for `access="read"` gets the strongest kind the
mounted tier can actually provide, and `repo_writable` records **which guarantee
was obtained** rather than which was requested. `False` means a tier is enforcing
it. Any wording here, in `ph doctor`, or in a config comment that blurs the two
is a defect (§12 Q10) — the whole point of the field is that a caller must not
have to infer a guarantee that is not there.

**There is always a workspace.** `acquire` never fails and never returns `None`:
an agent needs a working directory and somewhere to write, and "no workspace" is
not a state the lifecycle can be in. A provider that cannot serve a request —
`workspace-git-worktree` handed a `base` that is not a repository — *declines*,
and the seam falls back to `shared` with a logged notice. A hard failure there
would turn "this directory is not a git repo" into "pH will not start".

**A workspace is an effect of the scope that took it (I2).** `acquire` registers
its teardown through `ctx.effect`, which is the mechanism `Context.effect`'s own
docstring names a worktree for: a disposed agent scope unwinds the workspace
instead of leaving it held with nobody to release it. That is the *in-process*
half of cleanup; the `workspace/acquired`/`disposed` pair is the crash half,
reconciled at session open (§4.9), and a build that had only the second would
report a leak for every ordinary error path.

**`scratch` is always present and always writable**, on every kind and every
tier — and the *seam* creates it, so that guarantee has one implementation rather
than one per provider. It lives in pH's own state directory rather than in the
workspace, so it survives disposal as a session artifact: a child whose worktree
is discarded still leaves behind what it was asked to produce. A provider is
handed the path and may substitute its own — a sandbox tier has to put it
somewhere the container can reach — but never has to invent the layout.

@module ph.seams.workspace
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import anyio
from pydantic import Field

from ..cordis import Context, Disposer, maybe_await, plugin, safe_yaml_load
from ..paths import default_home_path
from ..session import Session
from ..wire import WireModel
from . import workspace_provision
from ._registry import claim_entry, claim_slot
from .diagnostics import Diagnostic, contribute
from .sandbox import SandboxPolicy
from .workspace_provision import ProvisionEntry, ProvisionReport

__all__ = [
    "PROJECT_PROVISION_FILE",
    "ContainmentTier",
    "DeclineReason",
    "LifecycleConfig",
    "SharedWorkspaceProvider",
    "Workspace",
    "WorkspaceAccess",
    "WorkspaceDeclined",
    "WorkspaceKind",
    "WorkspaceProvider",
    "WorkspaceSeam",
    "apply",
    "discover_provisioning",
    "fresh_root",
    "lifecycle",
    "project_access",
    "redirection_env",
    "workspace_of",
    "workspace_policy",
    "writable_roots",
]

log = logging.getLogger("ph.seams.workspace")

ContainmentTier: TypeAlias = Literal["advisory", "worktree", "sandbox"]
"""The ladder, named once.

`ph doctor` prints it (P4-12) and `containment.tier` selects it (P4-11); a
vocabulary spelled out at each of those sites is one where a typo reads as a
tier nobody has.
"""

WorkspaceKind: TypeAlias = Literal["shared", "worktree", "worktree-ephemeral", "readonly-scratch"]
"""What an agent actually got.

Four kinds rather than a boolean, because the interesting distinctions are not
writable-or-not. `shared` is today's behaviour — one checkout, no isolation.
`worktree` is that agent's own branch, merged back deliberately.
`worktree-ephemeral` is a full checkout the agent may write and whose writes
**reach nobody**: discarded on disposal, never merged. `readonly-scratch` is the
only one where the repository is genuinely unwritable, and only a sandbox backend
can deliver it.
"""

WorkspaceAccess: TypeAlias = Literal["write", "read"]
"""What the *caller* needs of `base` — a request, not a guarantee.

A research child asking for `read` should not be handed a checkout it might
mutate; but "read-only" is an enforcement claim, and the tier decides whether one
can be made. The answer comes back as `kind` and `repo_writable`.
"""


def project_access(kind: WorkspaceKind) -> WorkspaceAccess:
    """What a workspace of this kind grants of the **project** (E3).

    Not of the directory: `worktree-ephemeral` may be written freely and merges
    nothing, so what its holder was granted of the project is `read`. That
    distinction is `repo_writable`'s, read one level up, and it is what a spawn
    records as `granted_access` and what `ph doctor` prints per agent.

    Here rather than in the packages that ask, and exhaustive over `WorkspaceKind`
    rather than a membership test, so a kind added in this module cannot silently
    classify as `read` in `ph-rlm` — `mypy` refuses the match instead.
    """
    match kind:
        case "shared" | "worktree":
            return "write"
        case "worktree-ephemeral" | "readonly-scratch":
            return "read"


def fresh_root(kind: WorkspaceKind) -> bool:
    """Whether this kind hands the agent a directory that is not the base.

    Exhaustive over `WorkspaceKind` rather than `root == base`, for the reason
    `project_access` next door already gives: a kind added in this module should
    fail to type-check here rather than silently classify. `shared` is the only
    one whose root *is* the base, which is why nothing is provisioned into it —
    every material is already there, and copying `.env` onto itself is the one
    way this could destroy the file it exists to provide.
    """
    match kind:
        case "shared":
            return False
        case "worktree" | "worktree-ephemeral" | "readonly-scratch":
            return True


def redirection_env(scratch: Path) -> dict[str, str]:
    """Where the toolchain's droppings go instead of into the workspace (E12).

    Every entry is a cache or temp location a build tool writes *beside the
    sources* by default. Pointed inside `scratch`, which is outside the workspace
    and survives disposal, three things follow at once: a read-only repo becomes
    usable rather than merely safe, an ephemeral child's notes outlive the
    checkout that is thrown away, and — the one that matters at the `worktree`
    tier — `git status` reports the agent's work rather than `pytest`'s, so
    "remove a clean worktree, keep a dirty one" keeps meaning something.

    Beside `Workspace.env` rather than in the git tier that first needed it: the
    table is a property of *scratch*, not of worktrees, and §4.8 gives the same
    env to `readonly-scratch`. A copy in each provider is a copy that can drift.

    `PYTEST_ADDOPTS` disables the cache provider outright as well as moving
    `--basetemp`, because `.pytest_cache/` is written next to `rootdir` and no
    environment variable relocates it. `TMPDIR` is `scratch` itself: it must
    exist before the first `tempfile` call, and `scratch` is the one directory
    the seam guarantees.
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

    The one definition, because two consumers need the same set and a third
    arrives with P6-05: `permissions-fs`'s default rule prompts about what falls
    outside it, and `workspace_policy` below hands the same set to a backend to
    enforce. Two spellings that drifted would be exactly the defect §4.8's tier
    table exists to prevent — and a test asserting they agree catches drift
    rather than preventing it, which is why this is a function and not a
    convention. `Workspace.agent_work_pathspec` next door is the same move.

    `scratch` is in it, always. It is outside the worktree by design (E5) and is
    the one place a read-only or ephemeral agent is *told* it may write, so a
    scope naming only `root` would prompt on exactly the writes the design
    invites.
    """
    return (workspace.root, workspace.scratch)


def workspace_policy(workspace: Workspace) -> SandboxPolicy:
    """The workspace as a confinement request: write here, ask about elsewhere.

    On the seam, beside the value it describes, because two consumers need the
    *same* set — `ctx.shell` requests it of a backend, and `workspace-write-scope`
    prompts about what falls outside it. Two spellings of one boundary that
    drifted would be the defect §4.8's tier table exists to prevent, where a
    tier's name promises what its policy does not do.

    `scratch` is writable too, always: it is outside the worktree by design (E5)
    and is the one place a read-only agent is *told* it may write, so a policy
    that named only `root` would confine away the writes the design invites.
    """
    first, *extra = writable_roots(workspace)
    return SandboxPolicy(
        mode="workspace-write",
        workspace_root=str(first),
        writable_extra=[str(path) for path in extra],
    )


def workspace_of(ctx: Context, agent: Any) -> Workspace | None:
    """This agent's workspace, asked of a seam that may not be mounted.

    The question written once. Five callers had it — the prompt line, `bash`, the
    kernel, the spawn path and the fs resolver — and the copies had already
    disagreed about whether an agent with no `id` means `None` or a lookup with
    an empty key, and about whether a raising seam is fatal. `_registry`'s own
    docstring is about exactly this shape of drift.

    Fail-soft on purpose: a caller asking "where does this agent write" during a
    teardown, or in a profile that layers no workspace row, gets `None` and
    carries on with the process's own directory.
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
    friends inside `scratch`, because build tools write into the tree they are
    run against and an enforced-read-only repo would otherwise make "run the
    tests" impossible rather than merely safe. Best-effort by construction: a
    toolchain that insists on writing beside its sources will still fail, and the
    answer to that is `access="write"` for that agent, not a weaker tier.
    """
    provisioned: tuple[str, ...] = ()
    """Paths the seam put in this workspace (E14) — not the agent's work."""
    provision_failures: tuple[str, ...] = ()
    """Materials the seam could not put in place (E14).

    On the value rather than in a log line because the party that has to know
    `.env` is missing is the *agent* about to wonder why the tests fail — it is
    read straight onto the workspace prompt line. Empty is the ordinary case,
    including "this profile provisions nothing"."""
    release: Callable[[Workspace], Awaitable[bool]] | None = None
    """The provider's teardown, returning whether anything was **kept**.

    Takes the whole workspace, not one field of it: a teardown policy needs to
    know what was provisioned *and* what kind it is holding, and P4-09's
    checkpoint refs will be the third such fact — a callback that takes the value
    costs no protocol change when that happens.

    The answer is only knowable here: P4-08's policy is "keep dirty, remove
    clean, discard ephemeral even if dirty", so a field set at acquire time could
    not carry it and `kind` cannot be asked instead — a `worktree` is either. The
    seam records the answer on `workspace/disposed` so a reader can tell "nothing
    changed, so it was removed" from "these writes were thrown away by design".
    """

    def agent_work_pathspec(self) -> list[str]:
        """A `git` pathspec selecting this tree *minus* what the seam put in it.

        The one definition of "the agent's work", because there are already
        three consumers and they must not disagree: the disposal policy
        (`workspace_git._dirty`), `/workspaces list`, and P4-09's `/revert`,
        whose "restore tracked + untracked-not-ignored" is exactly the set that
        must not clobber a provisioned `node_modules`. Two of those answered it
        separately at first, and called the same tree clean and dirty.

        The positive `.` is required: exclusions alone match everything, which is
        the opposite of what they read as.
        """
        return [".", *(f":(exclude){entry}" for entry in self.provisioned)]


DeclineReason: TypeAlias = Literal[
    "not-a-repository", "branch-in-use", "path-exists", "provider-failed"
]
"""Why a tier could not serve a request, as a code rather than prose.

`ph doctor` prints it (P4-12) and the fallback is otherwise indistinguishable
from "no tier configured" — an operator who set `worktree` and got `shared` is
owed the reason, and a durable event carrying an English sentence is unparseable
by the consumer that has to branch on it.
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


@dataclass(frozen=True, slots=True)
class SharedWorkspaceProvider:
    """`workspace-shared` — today's behaviour, and the floor under every tier.

    Returns `base` itself, so mounting the seam changes nothing: no checkout, no
    copy, no cost. It is also the fallback the seam keeps for a provider that
    declines, which is why it lives beside the seam rather than in a row of its
    own — "there is always a workspace" is a promise the seam makes, and a
    promise kept by a row a profile might not layer is not one.

    `access="read"` is honoured by *saying so*: the kind stays `shared` and
    `repo_writable` stays `True`, because nothing here enforces anything. A
    provider that returned `repo_writable=False` for a shared checkout would be
    telling the caller a lie it would then act on.
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
    bookkeeping, and `Job.release` next door is the same split.
    """

    workspace: Workspace
    dispose: Disposer | None = None
    session: Session | None = None
    """Where this workspace's closing event goes, set once the acquisition is
    logged — the release closure reads it here rather than capturing it, so the
    two halves of the pair cannot disagree about which log they belong to."""


@dataclass(slots=True)
class WorkspaceSeam:
    """The service published as `ctx.workspace`."""

    ctx: Context
    shared: SharedWorkspaceProvider
    scratch_root: Path
    provider: WorkspaceProvider | None = None
    _provisioning: list[ProvisionEntry] = field(default_factory=list)
    """Materials to put in a fresh workspace, in registration order (E14).

    A list rather than a `claim_slot`, because two sources legitimately compose:
    the profile's own row, and a repository's `.ph-workspace.yml`. Later entries
    win a collision, which is the same last-write-wins a single list has.
    """
    _held: dict[str, _Held] = field(default_factory=dict)
    """Live workspaces by agent id.

    Held here rather than on the agent because the *question* is asked by things
    that have an agent id and no agent object — the prompt's workspace line, and
    `ph doctor`'s per-agent report. Emptied by the effect disposer, so an entry
    surviving its agent is the same leak `workspace/acquired` without a
    `disposed` records durably.
    """

    def of(self, agent_id: str) -> Workspace | None:
        """The workspace this agent holds, if it has acquired one.

        `None` is a real answer and the common one today: nothing acquires until
        the agent lifecycle does (P4-08), and a caller must say what is true
        rather than describing a workspace nobody took.
        """
        held = self._held.get(agent_id)
        return None if held is None else held.workspace

    def provision(
        self, entries: Sequence[ProvisionEntry], *, scope: Context | None = None
    ) -> Disposer:
        """Contribute materials for every fresh workspace this seam hands out.

        On the *seam* rather than on the tier, so `readonly-scratch` (P6-05) and
        any later fresh-root kind inherit the guards without re-implementing
        them — the same argument that put `scratch` here. Nothing is provisioned
        into a `shared` workspace, whose root *is* the base.

        Pass `scope=ctx` from a row's `apply`, so the entries leave with the row.
        Removal is by identity (`claim_entry`), because a `ProvisionEntry` is a
        *value*: two rows contributing `{source: .env}` compare equal, and
        `list.remove` would have one row's disposer take the other's.
        """
        disposers = [
            claim_entry(scope or self.ctx, self._provisioning, entry, label="workspace.provision")
            for entry in entries
        ]

        def release() -> None:
            for disposer in disposers:
                disposer()

        return (scope or self.ctx).add_disposer(release, label="workspace.provision")

    def live(self) -> list[Workspace]:
        """Every workspace an agent currently holds.

        Keyed lookups (`of`) answer "does *this* agent hold one"; this answers
        "is this tree anybody's", which is what `/workspaces` needs before it
        offers to delete a directory. Matching by root rather than by inverting
        a directory name back into an agent id is the point: `sanitize_ref` is
        lossy, so an id that does not sanitize to itself would read as unheld and
        lose the refusal that protects it.
        """
        return [held.workspace for held in self._held.values()]

    def register_provider(
        self, provider: WorkspaceProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim the tier. One at a time; `shared` remains the fallback."""
        return claim_slot(scope or self.ctx, self, "provider", provider, label="workspace.provider")

    def effective_tier(self, *, child: bool) -> ContainmentTier:
        """What one role actually gets, provider and choice reconciled.

        **Effective, not configured, in both directions.** A `worktree` row over
        a directory that is not a repository declines on every acquire, so a
        report reading the config would name containment nobody has — and since
        P4-11 the inverse is just as reachable: the shipped `rlm` profile layers
        the git provider *and* chooses `advisory` for the person's own agent, so
        reading the provider alone would name a worktree the root agent never
        gets. Both are the defect §4.8 closes on, from opposite sides.

        The two halves can each only *lower* the answer and neither can raise
        it: a chosen `advisory` declines a registered provider, and an absent
        provider cannot deliver whatever was chosen. `acquire` makes the same
        reconciliation by *doing* it, which is why this is the only other place
        allowed to state it — a third spelling would be a report that disagrees
        with the tree on disk, and `ph doctor` (P4-12) prints both roles because
        a deployment where they differ is the shipped one.
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

        **Per agent, not per profile**, because since P4-11 there is no single
        answer: the shipped `rlm` posture puts the person's own agent in their
        checkout and its children in worktrees, so a report naming one kind
        would be wrong about half the process. An agent that has acquired
        nothing prints nothing — `doctor` on an idle process legitimately has
        only the profile-level rows to show, and inventing a row per configured
        agent would describe workspaces nobody holds.
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
        return rows

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

        **The rung is derived, not asked of the caller** (P4-11). `shell.run`
        made the same call in writing — "the seam resolves it itself, rather
        than making each shell-shaped tool remember to" — and the reason is
        sharper here: a caller that forgets gets the provider, which for a
        *root* agent is the escalation the shipped profile says a deployment
        should have to ask for. The role is already in hand, because a child's
        session says so (`origin: "subagent"`), so nothing has to be passed and
        no third caller can forget.

        `tier` overrides that derivation for a caller who means something
        specific, exactly as `cwd` overrides `shell.run`'s. `advisory` declines
        a registered provider; anything else consults it.

        `scope` is what bounds the workspace's life — the agent's own scope, for
        the lifecycle that lands in P4-08. Disposing it releases the workspace
        and writes the closing event, so an error path that never reaches an
        explicit `dispose` is not a leak.
        """
        scratch = await self._scratch_for(session_id, agent_id)
        chosen = self._chosen_tier(session) if tier is None else tier
        workspace = None
        declined: DeclineReason | None = None
        if self.provider is not None and chosen != "advisory":
            try:
                workspace = await self.provider.acquire(
                    session_id=session_id,
                    agent_id=agent_id,
                    base=base,
                    scratch=scratch,
                    access=access,
                )
            except WorkspaceDeclined as refusal:
                # A decline that says why. Not an error path: half the
                # directories a person runs pH in are not repositories.
                declined = refusal.reason
                log.info(
                    "ph.seams.workspace: tier declined %s for agent %s (%s); using a shared "
                    "workspace, so this agent is not contained",
                    base,
                    agent_id,
                    refusal.reason,
                )
            except Exception:
                # A tier that broke is a tier that is not in force. Falling back
                # is the honest outcome and `workspace/acquired` will say
                # `shared`, which is what an operator needs to see.
                declined = "provider-failed"
                log.exception("ph.seams.workspace: provider failed; falling back to shared")
            else:
                if workspace is None:
                    # No reason *fabricated* here. A provider that declined
                    # without giving one has not told us why, and inventing
                    # a reason for it would reintroduce one level down the
                    # very confusion this field exists to remove — and would say
                    # "git" about a tier that may not be git at all.
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
        # Owned *before* provisioning, not after. Materialising a dependency
        # directory is thousands of syscalls, and running it between
        # `provider.acquire()` and the `ctx.effect` registration would put the
        # widest window in this module exactly where the worktree exists and
        # nothing yet unwinds it — against I2, in the module that argues I2.
        held = await self._track(workspace, agent_id, scope)
        held.workspace = await self._provision(workspace, base)
        self._log(held.workspace, agent_id, session, declined)
        return held.workspace

    async def _provision(self, workspace: Workspace, base: Path) -> Workspace:
        """Put the configured materials in a *fresh* root (E14)."""
        if not self._provisioning or not fresh_root(workspace.kind):
            return workspace
        # Qualified: `provision` on this class is the *registration* and the
        # module function is the work, and the module name is what keeps a call
        # site from doing the wrong one.
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

        "Early" because the scope owns it either way (I2); this is the same
        teardown reached deliberately rather than by unwinding, which is what
        `jobs.forget` and `subagents.delete` are to their own effects. Calling it
        twice is a no-op — the disposer deregisters itself.
        """
        held = self._held.get(agent_id)
        if held is None or held.dispose is None:
            return
        await maybe_await(held.dispose())

    async def _scratch_for(self, session_id: str, agent_id: str) -> Path:
        """Per session *and* per agent, created rather than merely named.

        Owned by the seam so the layout has one implementation: two children of
        one session writing notes into one directory is the collision this
        avoids, and a provider that got it wrong would break it for its tier
        alone with nothing checking.
        """
        scratch = self.scratch_root / session_id / agent_id
        await anyio.to_thread.run_sync(lambda: scratch.mkdir(parents=True, exist_ok=True))
        return scratch

    def _chosen_tier(self, session: Session | None) -> ContainmentTier | None:
        """Which rung this acquisition gets, read off the deployment's choice.

        The role comes from the session rather than from an argument: a child's
        header carries `origin: "subagent"` (the spawn stamps it), so "is this a
        child" is a fact the seam already holds. `None` — no containment row —
        means nobody chose, and a deployment that layered a provider and never
        mentioned containment gets that provider: layering it *was* the choice.

        A runtime `ctx.get` rather than an import, which is how every other
        cross-seam consult in this package is spelled — and it is what keeps the
        selector free to depend on this module's vocabulary without a cycle.
        """
        containment = self.ctx.get("containment")
        if containment is None:
            return None
        child = session is not None and session.header.origin == "subagent"
        chosen: ContainmentTier | None = containment.for_role(child=child)
        return chosen

    async def _track(self, workspace: Workspace, agent_id: str, scope: Context | None) -> _Held:
        """Register the teardown as an effect, so the workspace has an owner.

        The release closure reads `held.workspace` rather than capturing one,
        because provisioning replaces the value a moment later and the teardown
        policy needs the *final* one — what was put in the tree is what it must
        not mistake for the agent's work.
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
                    held.session.append(
                        "workspace/disposed", self._payload(current, agent_id, kept=kept)
                    )

            return release

        held.dispose = await (scope or self.ctx).effect(enter, label=f"workspace({agent_id})")
        return held

    def _log(
        self,
        workspace: Workspace,
        agent_id: str,
        session: Session | None,
        declined: DeclineReason | None,
    ) -> None:
        """Both halves of the durable pair are written by the seam.

        A pair only reconciles if one place owns both — a provider that forgot
        the second would leave every workspace looking leaked.
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
        session.append("workspace/acquired", data)
        if workspace.provision_failures:
            session.append(
                "workspace/provisioned",
                {"agentId": agent_id, "failed": list(workspace.provision_failures)},
            )

    def _payload(self, workspace: Workspace, agent_id: str, **extra: Any) -> dict[str, Any]:
        """The keys both halves of the pair share, spelled once.

        `ref` rides both so a reader can say which branch a turn ran against
        without inspecting the repository, and is omitted rather than sent as
        `null` for the kinds that have none.
        """
        data: dict[str, Any] = {"agentId": agent_id, **extra}
        if workspace.ref is not None:
            data["ref"] = workspace.ref
        return data


PROJECT_PROVISION_FILE = ".ph-workspace.yml"
"""Where a repository states what its worktrees need (E14).

Discovered the way `memory-agents-md` finds `AGENTS.md` — walking up from the
project directory — and read for **data only**: a `copy`/`symlink`/`hardlink`
entry, nothing that executes. That is what makes cloning a repository and
starting pH safe in a way `wtp`'s `.wtp.yml` is not, and it is why the `command`
hook was refused rather than trust-gated.

Read once, at mount. A file that appears later needs a restart, which is the
right trade for a list that is read before every agent starts.
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

    **The seam alone changes nothing; this row is what makes a tier bite.** It
    is separate from the seam's own row because the two answer different
    questions — "what happens when someone acquires" and "who acquires, and
    when" — and a deployment driving the lifecycle itself (a test, an embedder)
    wants the first without the second.

    Acquisition is *lazy and idempotent*, at the first `agent/pre-step`. Two
    reasons, and neither is convenience. `agent/created` is an `emit`, so a
    listener that has to `await git worktree add` could not hold the agent up
    and the first tool call would race the checkout. And a child's workspace is
    its parent's decision — base and `access` both — so the spawn path acquires
    first and this row must find that one rather than overwrite it, which
    `of()` already answers.
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

    Nearest-first and *first-wins*, which is `memory-agents-md`' rule: the file
    beside the code is the one that knows what the code needs, and a monorepo
    root should not override a package that states its own.

    Read with `safe_yaml_load`, the same reader every profile row goes through:
    this is the *least* trusted config the harness opens, so it is the last one
    that should have its own parsing rules — that reader rejects unknown `!tag`s
    ("pH config is data, not code") and the implicit timestamp coercion that
    would otherwise turn `source: 2024-01-02` into a `datetime` here and a `str`
    everywhere else.

    Every failure is a shrug: a malformed file, an entry with an unknown key, a
    `source` naming somewhere outside the tree. Refusing to start because a
    repository's optional config is wrong would make this list load-bearing,
    and the guards in `resolve_entry` refuse the dangerous ones per entry
    anyway.
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
